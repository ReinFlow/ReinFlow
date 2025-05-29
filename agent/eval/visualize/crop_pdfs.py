from pypdf import PdfReader, PdfWriter
def crop_pdf_any_percent(
    input_pdf_path,
    output_pdf_path,
    left_percent=0.0,
    right_percent=0.0,
    top_percent=0.0,
    bottom_percent=0.0
):
    """
    Crop a PDF by specified percentages from the left, right, top, and bottom of each page.
    
    Parameters:
    - input_pdf_path (str): Path to the input PDF file.
    - output_pdf_path (str): Path to save the cropped PDF file.
    - left_percent (float): Percentage of width to crop from the left (0.0 to 100.0).
    - right_percent (float): Percentage of width to crop from the right (0.0 to 100.0).
    - top_percent (float): Percentage of height to crop from the top (0.0 to 100.0).
    - bottom_percent (float): Percentage of height to crop from the bottom (0.0 to 100.0).
    
    Raises:
    - ValueError: If percentages are negative or total cropping exceeds page dimensions.
    - FileNotFoundError: If input PDF path does not exist.
    """
    # Validate percentages
    for percent, side in [
        (left_percent, 'left'),
        (right_percent, 'right'),
        (top_percent, 'top'),
        (bottom_percent, 'bottom')
    ]:
        if not 0.0 <= percent <= 100.0:
            raise ValueError(f"{side.capitalize()} percent must be between 0.0 and 100.0, got {percent}")

    if left_percent + right_percent >= 100.0:
        raise ValueError(f"Total left ({left_percent}%) and right ({right_percent}%) cropping exceeds 100% of page width")
    if top_percent + bottom_percent >= 100.0:
        raise ValueError(f"Total top ({top_percent}%) and bottom ({bottom_percent}%) cropping exceeds 100% of page height")

    # Open the input PDF
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    # Process each page
    for page in reader.pages:
        # Get the original media box (defines the page's dimensions)
        media_box = page.mediabox
        lower_left_x = float(media_box.lower_left[0])
        lower_left_y = float(media_box.lower_left[1])
        upper_right_x = float(media_box.upper_right[0])
        upper_right_y = float(media_box.upper_right[1])

        # Calculate page dimensions
        page_width = upper_right_x - lower_left_x
        page_height = upper_right_y - lower_left_y

        # Calculate cropping amounts
        crop_left = page_width * (left_percent / 100.0)
        crop_right = page_width * (right_percent / 100.0)
        crop_top = page_height * (top_percent / 100.0)
        crop_bottom = page_height * (bottom_percent / 100.0)

        # Calculate new coordinates
        new_lower_left_x = lower_left_x + crop_left
        new_lower_left_y = lower_left_y + crop_bottom
        new_upper_right_x = upper_right_x - crop_right
        new_upper_right_y = upper_right_y - crop_top

        # Update the media box
        page.mediabox.lower_left = (new_lower_left_x, new_lower_left_y)
        page.mediabox.upper_right = (new_upper_right_x, new_upper_right_y)

        # Update the crop box if it exists
        if page.get('/CropBox'):
            page.cropbox.lower_left = (new_lower_left_x, new_lower_left_y)
            page.cropbox.upper_right = (new_upper_right_x, new_upper_right_y)

        # Add the modified page to the writer
        writer.add_page(page)

    # Save the output PDF
    with open(output_pdf_path, 'wb') as output_file:
        writer.write(output_file)

   
if __name__ == "__main__":
    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Crop a PDF file by specified percentages.")
    parser.add_argument("--input_pdf", type=str, help="Path to the input PDF file")
    parser.add_argument("--output_pdf", type=str, help="Path to the output cropped PDF file")
    parser.add_argument("--left_percent", type=float, default=0.0, help="Percentage to crop from the left (default: 0.0)")
    parser.add_argument("--right_percent", type=float, default=0.0, help="Percentage to crop from the right (default: 0.0)")
    parser.add_argument("--top_percent", type=float, default=10.0, help="Percentage to crop from the top (default: 10.0)")
    parser.add_argument("--bottom_percent", type=float, default=4.0, help="Percentage to crop from the bottom (default: 4.0)")

    # Parse arguments
    args = parser.parse_args()

    # Call the crop function with parsed arguments
    crop_pdf_any_percent(
        input_pdf_path=args.input_pdf,
        output_pdf_path=args.output_pdf,
        left_percent=args.left_percent,
        right_percent=args.right_percent,
        top_percent=args.top_percent,
        bottom_percent=args.bottom_percent
    )
    print(f"Cropped PDF saved to {args.output_pdf}")
    