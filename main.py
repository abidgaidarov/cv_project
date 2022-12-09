import streamlit as st
import detection
import denoising_dirty_documents
import page_2

def main():
    
    page_names_to_funcs = {
    'Генерация цифр': page_2.generate_number,
    "Детекция автомобилей Тесла": detection.get_yolo_detection,
    "Восстановление документо": denoising_dirty_documents.clean_documents
    }
    
    selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
    page_names_to_funcs[selected_page]()

if __name__ == '__main__':
    main()