import os

# Ensure LibreOffice path is set for Docling
os.environ["DOCLING_LIBREOFFICE_CMD"] = r"C:\\Program Files\\LibreOffice\\program\\soffice.exe"
import time
import json
import tempfile
from pathlib import Path
import logging
import pandas as pd
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption, PowerpointFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc.document import DoclingDocument, PictureItem, TableItem
from docling_core.types.doc.base import ImageRefMode
from pdfconverter.convertword import convert_word
from pdfconverter.convertexcel import convert_excel
from pdfconverter.convertimage import convertimage
import pptxtopdf
import mimetypes
import tempfile
import shutil

# Configure logging (console only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configure pipeline options for PDF files
pdf_pipeline_options = PdfPipelineOptions()
pdf_pipeline_options.images_scale = 3.0  
pdf_pipeline_options.generate_picture_images = True
pdf_pipeline_options.do_table_structure = True

# Initialize DocumentConverter with format-specific options
doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
        InputFormat.DOCX: WordFormatOption(
            pipeline_options=PdfPipelineOptions(  # Word uses same pipeline options as PDF
                images_scale=3.0,
                generate_picture_images=True,
                do_table_structure=True
            )
        ),
        InputFormat.PPTX: PowerpointFormatOption(
            pipeline_options=PdfPipelineOptions(  # PowerPoint uses same pipeline options as PDF
                images_scale=3.0,
                generate_picture_images=True,
                do_table_structure=True
            )
        ),
    }
)

# Define supported file extensions for each converter
WORD_EXTENSIONS = {'.doc', '.docx', '.rtf', '.odt'}
EXCEL_EXTENSIONS = {'.xls', '.xlsx', '.ods', '.csv'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
POWERPOINT_EXTENSIONS = {'.ppt', '.pptx', '.odp'}
PDF_EXTENSIONS = {'.pdf'}

# All supported extensions
ALL_SUPPORTED_EXTENSIONS = (
    WORD_EXTENSIONS | EXCEL_EXTENSIONS | IMAGE_EXTENSIONS | 
    POWERPOINT_EXTENSIONS | PDF_EXTENSIONS
)

def get_file_extension(file_path):

    path = Path(file_path)
    
    # Primary method: use file extension from filename
    ext_from_name = path.suffix.lower()
    
    # Fallback method: use MIME type detection
    mime_type, _ = mimetypes.guess_type(str(path))
    ext_from_mime = None
    
    if mime_type:
        # Common MIME type to extension mappings
        mime_to_ext = {
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'application/vnd.ms-powerpoint': '.ppt',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx',
            'text/csv': '.csv',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/webp': '.webp'
        }
        ext_from_mime = mime_to_ext.get(mime_type)
    
    # Use filename extension if available and valid, otherwise use MIME type extension
    if ext_from_name and ext_from_name in ALL_SUPPORTED_EXTENSIONS:
        return ext_from_name
    elif ext_from_mime and ext_from_mime in ALL_SUPPORTED_EXTENSIONS:
        logging.info(f"Using MIME type detection for {file_path}: {ext_from_mime}")
        return ext_from_mime
    else:
        # Return the filename extension even if not in our supported list
        # This allows the calling code to handle unsupported files appropriately
        return ext_from_name if ext_from_name else ''

def is_supported_file(file_path):

    """Check if the file type is supported for processing. Currently skips PDF files."""
    ext = get_file_extension(file_path)
    # Skip PDF files for now
    # if ext == '.pdf':
    #     return False
    return ext in ALL_SUPPORTED_EXTENSIONS

def process_pdf(pdf_path, doc_output_dir):
    pdf_file = Path(pdf_path)
    if not pdf_file.is_file():
        logging.warning(f"PDF file not found, skipping: {pdf_path}")
        return

    start_time = time.time()
    conv_res = doc_converter.convert(str(pdf_file))
    doc_filename = doc_output_dir.name
        
    # Ensure output directory exists
    doc_output_dir.mkdir(parents=True, exist_ok=True)

    # Export images to the output directory
    image_dir = doc_output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    
    # Save images from the document
    image_count = 0
    for element, _ in conv_res.document.iterate_items():
        if isinstance(element, PictureItem):
            # Get the picture and save it
            picture = element.get_image(conv_res.document)
            if picture:
                # Sanitize the filename by replacing invalid characters
                safe_ref = element.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                image_filename = image_dir / f"{safe_ref}.png"
                
                # Ensure parent directory exists
                image_filename.parent.mkdir(parents=True, exist_ok=True)
                
                picture.save(image_filename)
                image_count += 1
                logging.debug(f"Saved image: {image_filename}")
    
    if image_count > 0:
        logging.info(f"Saved {image_count} images to {image_dir}")
    
    # Export markdown with referenced images
    content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
    
    # Save markdown
    md_filename = doc_output_dir / f"{doc_filename}.md"
    with md_filename.open("w", encoding="utf-8") as fp:
        fp.write(content_md)

    elapsed = time.time() - start_time
    logging.info(f"Converted {pdf_path} in {elapsed:.2f} seconds. Saved markdown to {md_filename}")


def process_non_pdf(input_path, doc_output_dir):
    """Process a non-PDF file directly with docling."""
    input_file = Path(input_path)
    if not input_file.is_file():
        logging.warning(f"Input file not found, skipping: {input_path}")
        return

    start_time = time.time()
    doc_filename = doc_output_dir.name

    # Ensure output directory exists
    doc_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save document as JSON using model_dump_json (native docling serialization)
    json_filename = doc_output_dir / f"{doc_filename}_DoclingDocument.json"

    # Try to load existing JSON
    conv_res = None

    if json_filename.exists():
        try:
            logging.info(f"Attempting to load existing DoclingDocument from JSON...")
            json_content = json_filename.read_text(encoding="utf-8")
            
            # Load the document using the new docling 2.57.0 API
            doc = DoclingDocument.model_validate_json(json_content)
            
            # Create a minimal wrapper to mimic ConversionResult
            # This avoids needing the full ConversionResult object
            class DocumentWrapper:
                def __init__(self, document):
                    self.document = document
            
            conv_res = DocumentWrapper(doc)
            logging.info(f"✓ Successfully loaded DoclingDocument from JSON")

            #check if the markdown file also exists
            md_filename = doc_output_dir / f"{doc_filename}.md"
            if not md_filename.exists():
                logging.info(f"Markdown file not found, will regenerate from loaded document.")
                
                # Export images
                image_dir = doc_output_dir / "images"
                image_dir.mkdir(exist_ok=True)
                for element, _ in conv_res.document.iterate_items():
                    if isinstance(element, PictureItem):
                        picture = element.get_image(conv_res.document)
                        if picture:
                            # Sanitize the filename
                            safe_ref = element.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                            image_filename = image_dir / f"{safe_ref}.png"
                            image_filename.parent.mkdir(parents=True, exist_ok=True)
                            picture.save(image_filename)
                
                content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
                with md_filename.open("w", encoding="utf-8") as fp:
                    fp.write(content_md)
                logging.info(f"Saved enhanced markdown to {md_filename}")

        except Exception as e:
            logging.warning(f"✗ Could not load JSON (likely old version): {type(e).__name__}")
            logging.info(f"Will reconvert document with docling 2.57.0...")
            conv_res = None
    
    # If loading failed or file doesn't exist, convert the document
    if conv_res is None:
        conv_res = doc_converter.convert(str(input_file))
        logging.info(f"✓ Converted {input_file.name} to DoclingDocument")

        # Save the document as JSON using model_dump_json
        try:
            with json_filename.open("w", encoding="utf-8") as fp:
                fp.write(conv_res.document.model_dump_json())
            logging.info(f"Saved document JSON to {json_filename}")
        except Exception as e:
            logging.warning(f"Could not save JSON file: {e}")

        # Export images to the output directory
        image_dir = doc_output_dir / "images"
        image_dir.mkdir(exist_ok=True)
        
        # Save images from the document
        image_count = 0
        for element, _ in conv_res.document.iterate_items():
            if isinstance(element, PictureItem):
                picture = element.get_image(conv_res.document)
                if picture:
                    # Sanitize the filename by replacing invalid characters
                    safe_ref = element.self_ref.replace('#/', '').replace('/', '_').replace('\\', '_').replace(':', '_')
                    image_filename = image_dir / f"{safe_ref}.png"
                    
                    # Ensure parent directory exists
                    image_filename.parent.mkdir(parents=True, exist_ok=True)
                    
                    picture.save(image_filename)
                    image_count += 1
                    logging.debug(f"Saved image: {image_filename}")
        
        if image_count > 0:
            logging.info(f"Saved {image_count} images to {image_dir}")
        
        # Export markdown with referenced images
        content_md = conv_res.document.export_to_markdown(image_mode=ImageRefMode.REFERENCED)
        
        # Save markdown
        md_filename = doc_output_dir / f"{doc_filename}.md"
        with md_filename.open("w", encoding="utf-8") as fp:
            fp.write(content_md)
        logging.info(f"Saved enhanced markdown to {md_filename}")

    elapsed = time.time() - start_time
    logging.info(f"Processed {input_path} in {elapsed:.2f} seconds")

    return conv_res


def process_file(input_path, output_root):
 
    input_file = Path(input_path)
    
    # Skip temporary Office files (start with ~$)
    if input_file.name.startswith('~$'):
        logging.info(f"Skipping temporary Office file: {input_file}")
        return False
    
    ext = get_file_extension(input_file)
    
    if ext in PDF_EXTENSIONS:

        convert_res = process_pdf(str(input_file), output_root / input_file.stem)

        return convert_res


    elif is_supported_file(input_file):

        convert_res = process_non_pdf(str(input_file), output_root / input_file.stem)
        
        return convert_res
        
def main():

    # Create log file in the output directory
    output_root = Path(r"C:\Users\uiv11567\source\repos\MESSelfService\Python\Analysis_Assistant\Test_Documents_Markdown")
    output_root.mkdir(parents=True, exist_ok=True)
    input_root = Path(r"C:\Users\uiv11567\source\repos\MESSelfService\Python\Test_Documents")

    # Validate input directory
    if not input_root.exists():
        logging.error(f"Input directory does not exist: {input_root}")
        return

    print(f"Starting batch conversion...")
    print(f"Input directory: {input_root}")
    print(f"Output directory: {output_root}")
    print(f"PDF Pipeline options:")
    print(f"  - do_picture_description: {pdf_pipeline_options.do_picture_description}")
    print(f"  - generate_table_images: {pdf_pipeline_options.generate_table_images}")
    print(f"  - generate_picture_images: {pdf_pipeline_options.generate_picture_images}")
    print(f"  - images_scale: {pdf_pipeline_options.images_scale}")

    processed_count = 0
    error_count = 0
    pdf_converted = []
    pdf_not_converted = []

    # Process all files recursively
    for file_path in input_root.rglob('*'):
        if file_path.is_file() and not file_path.name.startswith('~$'):
            print(f"\n{'='*60}")
            print(f"Processing file: {file_path}")
            print(f"{'='*60}")

            # Get the file path folder structure relative to input_root
            relative_path = file_path.relative_to(input_root).parent
            output_subdir = output_root / relative_path
            
            # try:
            converted = process_file(file_path, output_subdir)
            processed_count += 1
            

    print(f"\nBatch conversion completed.")
    print(f"Files processed: {processed_count}")
    print(f"Errors encountered: {error_count}")

    # Summary of PDF conversions
    print("\nPDF conversion summary:")
    logging.info("Processing complete.")

if __name__ == "__main__":
    main()