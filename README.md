# Automated Transaction Analysis
 Using Machine Learning this project attempts to perform Automated Transaction Analysis for use in Psychotherapy

This project utilizes the Cohn-Kanade dataset and you will be required to accept their licence agreement to obtain their data.
Download the cohn-kanade images, as well as the xls containing the annotations.

cohn-kanade.tgz
Cohn-Kanade Database FACS codes_updated based on 2002 manual_revised.xls


http://www.consortium.ri.cmu.edu/ckagree/

http://www.pitt.edu/~jeffcohn/biblio/Cohn-Kanade_Database.pdf



Upon obtaining the data, extract it alongside this README so that the image data lives in a folder called "cohn-kanade"
Download the XLS annotations alongside this README

Run the following


./run-all.sh


Human pose estimation


python classify.py

