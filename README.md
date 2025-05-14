# AI-DRIVEN-LIP-READING-SYSTEM-FOR-ASSISTIVE-COMMUNICATION
An intelligent system designed to empower individuals with speech impairments by converting silent lip movements into natural speech. This real-time solution leverages deep learning, computer vision, natural language processing, and speech synthesis to enable seamless assistive communication.

## Authors
- [@Rashaz Rafeeque](https://github.com/Rashaz-Raf)
- [@Jeevan A J](https://github.com/Jee-371)
- [@Rhishitha](https://github.com/rishi7736)

## Abstract
The potential of new assistive communication systems has been made possible by deep learning models' enhanced performance efficiency and precise lip-reading capabilities. People who have speech impairments use traditional communication methods which operate both slowly and with limited accessibility to produce ineffective communication outcomes. The research developed a deep learning system to convert spoken lip movements into audible voice signals for speech communication with people who have speech problems. The LipNet model architecture provides processing of silent video inputs through its Convolutional Neural Networks (CNNs) which perform feature extraction and Recurrent Neural Networks (RNNs) for sequence prediction through Connectionist Temporal Classification (CTC). Natural language processing technologies facilitate the text processing modules which identify both errors along with suitable contextual terms to generate precise and understandable transcriptions. The method shows remarkable potential to transform speech conversion into a faster and more accessible system that delivers accurate contextual results.

## System Overview
The AI-Driven Lip Reading System for Assistive Communication is designed to interpret lip movements from silent video input and convert them into natural, spoken language in real time. At its core, the system uses the LipNet deep learning model to accurately predict text from visual cues, capturing nuances in lip motion. Once the initial transcription is generated, natural language processing (NLP) techniques are applied to enhance the grammatical structure and semantic clarity of the output. This corrected text is then passed through a text-to-speech (TTS) engine, which vocalizes the message, effectively giving a voice to users who are unable to speak. The entire pipeline is wrapped in a user-friendly Streamlit interface, ensuring easy interaction and seamless performance. The system is particularly valuable for individuals with speech impairments, providing them with a real-time, AI-powered communication tool.

<img src="https://github.com/Rashaz-Raf/AI-DRIVEN-LIP-READING-SYSTEM-FOR-ASSISTIVE-COMMUNICATION/blob/main/WORKSPACE/IMAGES/Graphical_Abstract.png" alt="Methodology Flowchart" style="width: 50%;"/>

## Problem Definition 
People with speech impairments often struggle to communicate effectively, relying on slow or limited assistive tools. Traditional methods like text boards or sign language may not always be practical or accessible. This project aims to address these limitations by developing a system that reads lip movements and converts them into clear, spoken language. The key challenge lies in accurately interpreting silent visual speech and generating natural audio output in real time. This solution offers a more intuitive, efficient, and autonomous way for users to communicate in everyday scenarios.

## Methodology
The system workflow consists of the following key stages:

1. **Video Input Capture**: Users record short video clips of their lip movements using a camera.
2. **Lip Reading with LipNet**: The video is processed through the LipNet model, which uses spatiotemporal convolutions and CTC (Connectionist Temporal Classification) decoding to predict character sequences from visual lip movements.
3. **Text Correction with NLP**: The raw transcriptions are corrected and structured using pre-trained NLP models to improve sentence fluency and grammar.
4. **Speech Synthesis with TTS**: The corrected text is converted to speech using a TTS engine (e.g., pyttsx3 or gTTS), allowing real-time auditory output.

### Methodology Flowchart
<img src="https://github.com/Rashaz-Raf/AI-DRIVEN-LIP-READING-SYSTEM-FOR-ASSISTIVE-COMMUNICATION/blob/main/WORKSPACE/IMAGES/Methodology_Flowchart.png" alt="Methodology Flowchart" style="width: 50%;"/>

## Results
The system has demonstrated accurate lip-reading performance in controlled environments and successfully vocalized coherent sentences. Sample outputs include:

### Results 1
**Lip Input :**  “HELLO HOW ARE YOU”
**Corrected Text :** “Hello, how are you?”
**Synthesized Speech:** Played using TTS engine.

### Results 2
**Lip Input :**  “I NEED SOME WATER”
**Corrected Text :** “I need some water.”
**Synthesized Speech:** Played using TTS engine.

These results highlight the system’s ability to produce intelligible and natural-sounding speech from silent video clips.

## Features
- Real-time lip reading and speech synthesis
- User-friendly interface
- Supports multiple sentences and punctuation correction
- Works offline using pre-trained models
- Modular and scalable design for future enhancements

## Technologies Used
- Python
- TensorFlow / PyTorch (LipNet)
- OpenCV
- NLTK / spaCy / Transformers (for NLP correction)
- pyttsx3 / gTTS (for text-to-speech synthesis)
- Streamlit (for UI interface)

## Conclusion
This project marks a significant advancement in assistive technology by enabling silent speech communication. Through the fusion of lip reading, NLP, and speech synthesis, it offers a seamless and scalable solution for empowering users with speech impairments to communicate naturally and independently.

Future work will focus on improving robustness to varying lighting and angles, supporting continuous lip movement capture, enabling multilingual speech synthesis, and integrating with wearable devices (e.g., smart glasses).

## Code
The full codebase for the **DYNAMIC NARRATIVES** is available [here](https://github.com/Rashaz-Raf/AI_Powered_Video_Synthesis/tree/main/Source_Code).

## Documentation
Refer to the complete project documentation for detailed insights into the methodology, implementation, and results.

### Project Report Documentation
[Project Report Final PDF](https://github.com/Rashaz-Raf/AI_Powered_Video_Synthesis/blob/main/PROJECT_REPORT.pdf)
