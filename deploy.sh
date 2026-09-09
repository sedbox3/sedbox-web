#!/bin/bash
cd /home/container
git clone https://github.com/sedbox3/sedbox-web.git .
pip install -r requirements.txt
python web/app.py
