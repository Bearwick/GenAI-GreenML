import os
import random
from datetime import datetime

def generate_mock_email(email_id, email_type):
    """Generate a mock email (genuine or spam) and save as .eml."""
    if email_type == 'good':
        senders = ['friend@example.com', 'colleague@university.edu', 'family@gmail.com', 'support@bank.com']
        subjects = ['Meeting Tomorrow', 'Project Update', 'Catch Up Soon', 'Account Confirmation']
        bodies = [
            'Hi Femi,\nLet’s meet tomorrow at 10 AM to discuss the project.\nRegards,\nJohn',
            'Hello Femi,\nThe report is ready. Can we review it this week?\nBest,\nSarah',
            'Hey Femi,\nHow about dinner this weekend?\nCheers,\nMike',
            'Dear Femi,\nYour account has been updated. Contact us at support@bank.com.\nThanks,\nTeam'
        ]
    else:  # spam
        senders = ['win@lottery.com', 'offer@deals.xyz', 'prize@free.com', 'alert@secure.xyz']
        subjects = ['Claim Your Prize Now!', 'You Won $1000!', 'Urgent: Act Now!', 'Verify Your Account!']
        bodies = [
            'Dear User,\nClick here: http://fake.deals/claim to win $1000!\nHurry!\nBest,\nTeam',
            'Congratulations!\nYou’ve won a free gift. Visit http://scam.xyz now!\nCheers,\nAdmin',
            'Urgent!\nYour account needs verification. Go to http://phish.me/verify.\nThanks,\nSupport',
            'Win a free iPhone! Click http://offer.deals/promo to claim now!\nBest,\nPromo Team'
        ]
    
    email_content = f"""From: {random.choice(senders)}
To: femi@example.com
Subject: {random.choice(subjects)}
Date: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S +0100')}
Content-Type: text/plain

{random.choice(bodies)}"""
    
    # Create directory
    directory = f'emails/{email_type}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save email
    with open(f'{directory}/email_{email_id}.eml', 'w', encoding='utf-8') as f:
        f.write(email_content)
    print(f"Generated {email_type}/email_{email_id}.eml")

def main():
    """Generate 250 genuine and 250 spam emails."""
    for i in range(1, 251):
        generate_mock_email(i, 'good')
        generate_mock_email(i, 'spam')

if __name__ == '__main__':
    main()
