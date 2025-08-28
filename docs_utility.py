#docs_utility
import os
import zipfile
from lxml import etree
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
from openai import AzureOpenAI
import azure_config_list # your config with model_details dict

model_details = azure_config_list.model_details
model_data=model_details
def explain_table_with_llm(table_details, model_data):    
    # Example implementation for LLM integration
    explanation_prompt = f"Explain the following table:\n\n{table_details}\n\n.Give output in paragraphs only,not in points"
    try:
    # model_data = get_azure_llm_model_details(model_data)
        client = AzureOpenAI(
            api_key=model_data['api_key'],
            azure_deployment=model_data['deployment_name'],
            api_version=model_data['api_version'],
            azure_endpoint=model_data['base_url']
        )
        # Create and send the request
        response = client.chat.completions.create(
            model= model_data['deployment_name'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": explanation_prompt
                    }
                ]}
        ],
            max_tokens=2000
        )
        
        response = response.choices[0].message.content

        return response
    except Exception as e:
        print(f"Error explaining the table: {str(e)}")


def encode_image(image_path):
    try:
        """Encode an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f" Error encoding image: {str(e)}.")

def explain_image_with_llm(encoded_image, model_data):
    try:
        # prompt = f"Describe every detail present in the image. URL = data:image/jpeg;base64,{encoded_image}"
        prompt = [{
                "type": "text",
                "text": "Describe every detail present in the image.Do not give output in points.Give in paragraphs only"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}"
                }
            }]
        
        # model_data = get_azure_llm_model_details(model_data)
        client = AzureOpenAI(
            api_key=model_data['api_key'],
            azure_deployment=model_data['deployment_name'],
            api_version=model_data['api_version'],
            azure_endpoint=model_data['base_url']
        )
        # Create and send the reques
        response = client.chat.completions.create(
            model= model_data['deployment_name'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )

        # Append the response to the combined responses
        image_response = response.choices[0].message.content
        return image_response
    except Exception as e:
       
        print(f"ERROR converting image to text: {str(e)}.")


def process_docx_file(docx_file, output_dir, model_data):
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(docx_file, 'r') as zip_ref:
        document_xml = zip_ref.read('word/document.xml')
        tree = etree.XML(document_xml)

        media_folder = os.path.join(output_dir, "media")
        os.makedirs(media_folder, exist_ok=True)

        # Extract images
        for file_name in zip_ref.namelist():
            if file_name.startswith("word/media/"):
                with zip_ref.open(file_name) as file:
                    image_data = file.read()
                    image_path = os.path.join(media_folder, os.path.basename(file_name))
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_data)

        namespaces = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
        }

        content_text = []
        image_counter = 1
        table_counter = 1

        def get_image_path(counter):
            for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.gif']:
                candidate = os.path.join(media_folder, f"Image{counter}{ext}")
                if os.path.exists(candidate):
                    return candidate
            return None

        for element in tree.iter():
            tag_name = etree.QName(element).localname

            if tag_name == 'tbl':
                table_content = []
                for row in element.findall('.//w:tr', namespaces):
                    row_data = []
                    for cell in row.findall('.//w:tc', namespaces):
                        cell_text_parts = []
                        for para in cell.findall('.//w:p', namespaces):
                            texts = [node.text or "" for node in para.iter() if etree.QName(node).localname == 't']
                            cell_text_parts.append("".join(texts).strip())
                        cell_text = "\n".join(cell_text_parts).strip()
                        row_data.append(cell_text)
                    table_content.append(row_data)

                table_details = "\n".join(
                    " | ".join(row) for row in table_content if any(row)
                )
                table_explanation = explain_table_with_llm(table_details, model_data)
                content_text.append(table_explanation)
                table_counter += 1
                continue

            if tag_name == 'p':
                parent = element.getparent()
                while parent is not None:
                    if etree.QName(parent).localname == 'tc':
                        break
                    parent = parent.getparent()
                else:
                    texts = [node.text or "" for node in element.iter() if etree.QName(node).localname == 't']
                    para_text = "".join(texts).strip()
                    if para_text:
                        content_text.append(para_text)

            if tag_name == 'drawing':
                image_path = get_image_path(image_counter)
                if image_path:
                    encoded_image = encode_image(image_path)
                    explanation = explain_image_with_llm(encoded_image, model_data)
                    content_text.append(explanation)
                image_counter += 1

        return "\n\n".join(content_text)


def load_docx_documents(folder_path, model_data, temp_output_dir="temp_docx_processing"):
    documents = []
    doc_paths = []

    print(f"Loading DOCX documents from: {folder_path}")

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith('.docx'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing DOCX: {file_name}")
            try:
                extracted_text = process_docx_file(file_path, temp_output_dir, model_data)
                documents.append(extracted_text)
                doc_paths.append(file_name)
                print(f"Loaded DOCX: {file_name} ({len(extracted_text)} characters)")
            except Exception as e:
                print(f"Failed to process DOCX {file_name}: {e}")

    if not documents:
        raise ValueError("No DOCX documents were loaded. Ensure the folder contains valid DOCX files.")

    print(f"Total DOCX documents loaded: {len(documents)}")
    return documents, doc_paths


def split_text_into_chunks(text, chunk_size=200, chunk_overlap=30):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_text(text)
    print(f"Text split into {len(chunks)} chunks.")
    return chunks



folder_path=r'C:\Users\ashish.i.choudhary\Latest Code\rag\documents'
documents,paths=load_docx_documents(folder_path,model_data)
print(type(documents))
print(documents)

