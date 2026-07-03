import asyncio
import logging
from io import BytesIO

from aiogram import Bot, Router, F
from aiogram.types import Message, BufferedInputFile
from aiogram.filters import CommandStart, Command
from aiogram.exceptions import TelegramBadRequest

from pdf_builder import images_to_pdf

logger = logging.getLogger(__name__)

router = Router()

# Buffer for collecting media group messages: {media_group_id: [Message, ...]}
_media_groups: dict[str, list[Message]] = {}
# Pending tasks for media group finalization
_media_group_tasks: dict[str, asyncio.Task] = {}
# Users in scan mode: {user_id: True}
_scan_mode: dict[int, bool] = {}
# Scan mode for media groups: {media_group_id: True}
_media_group_scan: dict[str, bool] = {}

MEDIA_GROUP_TIMEOUT = 2.0

IMAGE_MIME_PREFIXES = ("image/jpeg", "image/png", "image/bmp", "image/tiff", "image/webp", "image/gif")


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я конвертирую фото в PDF.\n\n"
        "Отправь мне одно фото — получишь PDF.\n"
        "Отправь несколько фото пачкой — получишь один PDF со всеми фото.\n"
        "Можно отправлять как сжатое фото, так и файлом (документом).\n\n"
        "/scan — режим сканера (выравнивание + контраст для документов)"
    )


@router.message(Command("scan"))
async def cmd_scan(message: Message) -> None:
    uid = message.from_user.id
    if _scan_mode.get(uid):
        _scan_mode.pop(uid, None)
        await message.answer("Режим сканера выключен. Фото пойдут как есть.")
    else:
        _scan_mode[uid] = True
        await message.answer(
            "Режим сканера включён!\n"
            "Теперь фото будут обработаны: выравнивание перспективы + улучшение контраста.\n"
            "Отправь /scan ещё раз чтобы выключить."
        )


async def _download_photo(message: Message, bot: Bot) -> bytes | None:
    """Download the best quality photo or document image from a message."""
    try:
        if message.photo:
            photo = message.photo[-1]
            logger.info("Downloading photo file_id=%s (%dx%d)", photo.file_id[:12], photo.width, photo.height)
            file = await bot.download(photo)
            return file.read()
        if message.document and message.document.mime_type:
            if message.document.mime_type.startswith("image/"):
                logger.info("Downloading document file_id=%s mime=%s", message.document.file_id[:12], message.document.mime_type)
                file = await bot.download(message.document)
                return file.read()
    except TelegramBadRequest as e:
        if "file is too big" in str(e):
            logger.warning("File too big, skipping (>20MB): %s", e)
            await message.answer("Файл слишком большой (>20 МБ). Telegram не позволяет ботам скачивать такие файлы. Отправьте сжатым фото, а не документом.")
        else:
            logger.error("TelegramBadRequest while downloading: %s", e)
        return None
    return None


async def _finalize_media_group(media_group_id: str, bot: Bot) -> None:
    """Wait for all messages to arrive, then download photos and build PDF."""
    await asyncio.sleep(MEDIA_GROUP_TIMEOUT)

    messages = _media_groups.pop(media_group_id, [])
    _media_group_tasks.pop(media_group_id, None)
    scan = _media_group_scan.pop(media_group_id, False)

    if not messages:
        return

    chat_id = messages[0].chat.id

    try:
        # Download all photos now (after collecting all messages)
        image_list: list[bytes] = []
        for msg in messages:
            data = await _download_photo(msg, bot)
            if data:
                image_list.append(data)

        if not image_list:
            logger.warning("Media group %s: no images downloaded", media_group_id)
            await bot.send_message(chat_id, "Не удалось скачать ни одного фото из пачки.")
            return

        logger.info("Media group %s: building PDF from %d photos (scan=%s)", media_group_id, len(image_list), scan)
        pdf_data = images_to_pdf(image_list, scan=scan)
        pdf_file = BufferedInputFile(pdf_data, filename="photos.pdf")
        await bot.send_document(chat_id, pdf_file, caption=f"PDF из {len(image_list)} фото")
        logger.info("Media group %s: PDF sent (%d bytes)", media_group_id, len(pdf_data))
    except Exception as e:
        logger.error("Failed to process media group %s: %s", media_group_id, e)
        await bot.send_message(chat_id, "Ошибка при создании PDF. Попробуйте ещё раз.")


async def _handle_image_message(message: Message, bot: Bot) -> None:
    """Common handler for photo and document-image messages."""
    scan = _scan_mode.get(message.from_user.id, False)

    # Media group — collect messages and defer (download later)
    if message.media_group_id:
        mg_id = message.media_group_id
        if mg_id not in _media_groups:
            _media_groups[mg_id] = []
        if scan:
            _media_group_scan[mg_id] = True
        _media_groups[mg_id].append(message)
        count = len(_media_groups[mg_id])
        logger.info("Media group %s: added message #%d from user %s (scan=%s)", mg_id, count, message.from_user.id, scan)

        # Cancel previous timer and start a new one
        old_task = _media_group_tasks.get(mg_id)
        if old_task:
            old_task.cancel()
        _media_group_tasks[mg_id] = asyncio.create_task(_finalize_media_group(mg_id, bot))
        return

    # Single photo — download and send immediately
    logger.info("Single photo from user %s (scan=%s)", message.from_user.id, scan)
    image_data = await _download_photo(message, bot)
    if image_data is None:
        return

    try:
        pdf_data = images_to_pdf([image_data], scan=scan)
        pdf_file = BufferedInputFile(pdf_data, filename="photo.pdf")
        await message.answer_document(pdf_file, caption="Готово!")
    except Exception as e:
        logger.error("Failed to create PDF: %s", e)
        await message.answer("Ошибка при создании PDF. Попробуйте ещё раз.")


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    await _handle_image_message(message, bot)


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        await _handle_image_message(message, bot)
