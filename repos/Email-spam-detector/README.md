# Email Spam Detector
An Internship Project.

## Overview
This project generates 500 emails, both good and spam, and creates a folder for each category. It then detects whether each email is good or spam, extracts the URLs, and creates a CSV file containing their classifications using Python and Naive Bayes. It extracts URLs as a feature.

## Files
- `generate_emails.py`: Generates 500 mock .eml files.
- `spam_detector.py`: Classifies emails and extracts URLs.
- `email_analysis.csv`: Output with email data and predictions.
- `requirements.txt`: Dependencies.
- `emails/`: Contains generated emails.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run `generate_emails.py` to create emails.
3. Run `spam_detector.py` to classify emails and generate `email_analysis.csv`.

## Results
Achieved 100% accuracy on the test set.

## Notes
- Emails were generated randomly.
- The spam detection model uses TF-IDF features and Naive Bayes.
- The full emails/ folder is available upon request


