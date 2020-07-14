"""Main module for the streamlit app"""
import streamlit as st
import awesome_streamlit as ast

import src.pages.about
import src.pages.demo
import src.pages.hands_on


PAGES = {
    'About Project': src.pages.about,
    'Demo Burned Areas Detection': src.pages.demo,
    'Hands-on Burned Areas Detection': src.pages.hands_on
}


def main():
    """Main function of the app"""
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio('Go to', list(PAGES.keys()))

    st.sidebar.title("Code")
    st.sidebar.info(
        "Code is available at [GitHub](https://github.com/rpartsey/python-architecture-project). "
        "Checkout 'gcp_deploy' branch"
    )

    page = PAGES[selection]

    ast.shared.components.write_page(page)


main()