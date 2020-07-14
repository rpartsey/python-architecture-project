"""About page shown when the user enters the application"""
import streamlit as st
import awesome_streamlit as ast


def write():
    """Writes content to the app"""
    st.title('[Burned Areas Detection (BAD) Project](https://apps.ucu.edu.ua/en/ruslan-partsey/)')
    st.write("""
    **Problem statement:** 
    Hundreds of thousands of hectares of natural areas are destroyed in Ukraine every year by wildfires. 
    This natural disaster causes huge and sometimes irreparable damage to nature. Natural vegetation is 
    replaced by weeds resistant to regular burnout. Animals and plants listed in the Red Data Book of Ukraine 
    and international lists of protected species die in fire. It takes a lot of time for natural ecosystems 
    to recover and some of their components can not be recovered without human assistance. It takes hours for 
    experts to analyze a satellite image (experts spend about 20 hours to label an image  of 8000 by 8000 
    pixels covering 25 square kilometers of land). It is extremely expensive and labor-intensive to monitor 
    vast land areas.
    
    **Proposed solution:**
    Our project proposes to apply AI to automatically detect and pixel-wise classify burned areas on satellite images. 
    The peculiarity of our approach is the ability to work with heterogeneous multi-temporal satellite images, 
    thus suggesting a more effective method for wildfire damage estimation and analysis.
    The Machine Learning part of the project will be based on the remote sensing change detection algorithms. 
    Specifically, we will develop and deploy an AI pipeline that can automatically identify burned areas on the 
    uploaded heterogeneous multi-temporal satellite images. Also, to make it possible to integrate our application 
    with other programs/applications we will create a public API.
    
    **Value:**
    All research results will be openly accessible and could be used by a community of users. 
    An application automatically processing satellite images could also be used to monitor any other type of 
    Earth’s surface changes (from natural disasters to field anomalies in agriculture) and their impact on the climate.

    ### Hands-on Burned Areas Detection Demo
    """
    )
    ast.shared.components.video_youtube(
        src="https://www.youtube.com/embed/6Z2uYhO3gcU"
    )
