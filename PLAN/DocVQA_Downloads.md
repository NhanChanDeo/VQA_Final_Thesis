# Downloads - Document Visual Question Answering (DocVQA)

> **Source Portal:** [Robust Reading Competition (RRC) - Challenge 17](https://rrc.cvc.uab.es/?ch=17&com=downloads)  
> **Direct Dataset Host:** `https://datasets.cvc.uab.es/rrc/DocVQA/`  
> *Note:* Both the official portal redirect URLs and the decoded direct backend URLs are provided below.

---

## Terms and Conditions

By downloading the image data and annotations from the URLs below, you agree to the following terms and conditions:

1. You will **NOT** distribute the download URLs.
2. The Computer Vision Center (CVC) and IIIT Hyderabad make no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
3. You accept full responsibility for your use of the data and shall defend and indemnify Computer Vision Center (CVC) and IIIT Hyderabad including their employees, officers, and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.
4. All the images in **DocVQA - Single Page Document VQA task** are pages of documents downloaded from [UCSF Industry Documents Library](https://www.industrydocuments.ucsf.edu/). Documents are from various collections spanning different industries such as tobacco, food, drug, etc. Most of these documents' copyright is with the respective companies/trusts. The data is hosted purely for non-commercial, research, and educational purposes only. You may read UCSF Industry Documents Library's copyright and fair use policy [here](https://www.industrydocuments.ucsf.edu/help/copyright/).
5. All the images provided for the **DocVQA - Document Collection VQA task** are sourced from the [Public Disclosure Commission (PDC)](https://www.pdc.wa.gov/) documents. Information provided on the PDC website is considered public information and may be distributed or copied. Document images are offered purely for non-commercial research and educational purposes only. Downloading entails that you have read and agree with the policy of PDC as described [here](https://www.pdc.wa.gov/privacy-notice).
6. All images in **DocVQA - Infographics VQA task** are infographic images sourced from the internet. The URLs are shared so researchers can download the images and use them along with annotations for question-answers purely for non-commercial, research, and education purposes.
7. The annotations offered along with the document images are offered without any restrictions.

---

## Download Links & Tasks Hierarchy

### Task 4 - Multipage Document Visual Question Answering (MP-DocVQA)

*The dataset for Multipage DocVQA is available to download from the following URLs:*

* **V1.0 (uploaded on 19 February 2023):**  
  Consists of 46,436 questions posed over 5,929 documents with 47,952 pages in total. Besides questions and answers, document page images and OCR extracted with **Amazon Textract OCR** of all pages (64,057 pages) that belong to documents within the MP-DocVQA dataset are provided, skipping the 20-page limit.
  * **Questions and Answers:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task4/qas.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazQvcWFzLnppcA==)
  * **Images:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task4/images.tar.gz)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazQvaW1hZ2VzLnRhci5neg==)
  * **OCR results (Amazon Textract):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task4/ocr.tar.gz)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazQvb2NyLnRhci5neg==)
  * **IMDBs (processed dataset for [MP-DocVQA framework](https://github.com/rubenpt91/MP-DocVQA-Framework)):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task4/mpdocvqa_imdbs.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazQvbXBkb2N2cWFfaW1kYnMuemlw)

---

### Task 1 - Single Page Document Visual Question Answering (SP-DocVQA)

*Dataset for Single Page Document VQA (SP-DocVQA) task:*

* **[20 April 2020] Train V1.0:** Dataset released comprising 50,000 questions framed on 12,767 document images.
* **[31 August 2023] Updated:** Updated to include the training and validation question types.
  * **Annotations (questions, answers, question types...):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_qas.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfcWFzLnppcA==)
  * **Images:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_images.tar.gz)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfaW1hZ2VzLnRhci5neg==)
  * **OCR (Microsoft OCR):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_ocr.tar.gz)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfb2NyLnRhci5neg==)
  * **IMDBs (processed dataset for [MP-DocVQA framework](https://github.com/rubenpt91/MP-DocVQA-Framework)):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task1/spdocvqa_imdb.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazEvc3Bkb2N2cWFfaW1kYi56aXA=)

---

### Task 2 - Document Collection Visual Question Answering (DocCVQA)

*Defined over a collection of documents, over which natural language questions (queries) are defined. Provides a series of sample queries along with the relevant documents for each one.*

* **V0.1 (March 30, 2020):** Comprises 14,362 document images, 8 sample queries, and list of relevant responses (document IDs) for each sample query.
* **V0.2 (April 23, 2020):** Format of candidate names adjusted to 'Name Surname'. Test queries made public.
  * **Image Collection:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task2_images.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazJfaW1hZ2VzLnppcA==)
  * **Sample queries and responses:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/sample_public_v02.json)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvc2FtcGxlX3B1YmxpY192MDIuanNvbg==)
  * **Test queries:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/test_public.json)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvdGVzdF9wdWJsaWMuanNvbg==)
* **Evaluation Scripts:**
  * **DocCVQA Evaluation scripts:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task2%20evaluation%20script.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazIlMjBldmFsdWF0aW9uJTIwc2NyaXB0LnppcA==)

---

### Task 3 - Infographics VQA

*Participants download images using URLs provided in the JSON file. OCR outputs provided using Amazon Textract OCR system as auxiliary data.*

* **[10 Nov 2020] Train V0.1:** Training subset with around 10K questions defined on ~2,000 images.
* **[23 Dec 2020] Train 1.0:** 23,946 questions; 4,406 images.
* **[05 Jan 2021] Val 1.0:** 2,801 questions; 500 images.
* **[11 Feb 2021] Test 1.0:** 3,288 questions; 579 images.
* **[31 August 2023] Updated:** Includes **question types** in the validation set.
  * **Annotations (questions, answer, question types...):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_qas.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMvaW5mb2dyYXBoaWNzdnFhX3Fhcy56aXA=)
  * **OCR (Amazon Textract):**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_ocr.tar.gz)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMvaW5mb2dyYXBoaWNzdnFhX29jci50YXIuZ3o=)
  * **Images:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task3/infographicsvqa_images.tar.gz)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMvaW5mb2dyYXBoaWNzdnFhX2ltYWdlcy50YXIuZ3o=)
* **Evaluation Scripts:**
  * **Infographics VQA Evaluation scripts:**
    * [Direct URL](https://datasets.cvc.uab.es/rrc/DocVQA/Task3%20evaluation%20script.zip)
    * [RRC Portal Link](https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMlMjBldmFsdWF0aW9uJTIwc2NyaXB0LnppcA==)
