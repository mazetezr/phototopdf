from io import BytesIO

from PIL import Image

from scanner import scan_document


def _prepare_image(data: bytes, scan: bool = False) -> Image.Image:
    """Convert image bytes to RGB PIL Image. Optionally scan document first."""
    if scan:
        data = scan_document(data)
    img = Image.open(BytesIO(data))
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1])
        return background
    return img.convert("RGB")


def images_to_pdf(image_bytes_list: list[bytes], scan: bool = False) -> bytes:
    """Convert a list of image bytes into a single PDF document."""
    if not image_bytes_list:
        raise ValueError("No images provided")

    images = [_prepare_image(data, scan=scan) for data in image_bytes_list]

    pdf_buffer = BytesIO()
    if len(images) == 1:
        images[0].save(pdf_buffer, format="PDF")
    else:
        images[0].save(pdf_buffer, format="PDF", save_all=True, append_images=images[1:])
    pdf_buffer.seek(0)
    return pdf_buffer.read()
