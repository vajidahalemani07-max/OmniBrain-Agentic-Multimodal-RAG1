import pymupdf
def render_page(page,page_number):
    #render a pdf page as png image
    pix=page.get_pixmap(matrix=pymupdf.Matrix(2,2))
    image_path=f"temp_page_{page_number}.png"
    pix.save(image_path)
    return image_path