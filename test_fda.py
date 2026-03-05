import requests
import urllib.parse

def test_interaction(drug1, drug2):
    q1 = f'(openfda.generic_name:"{drug1}"+AND+drug_interactions:"{drug2}")'
    q2 = f'(openfda.generic_name:"{drug2}"+AND+drug_interactions:"{drug1}")'
    query = f'{q1}+OR+{q2}'
    url = f'https://api.fda.gov/drug/label.json?search={query}&limit=1'
    
    print(url)
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get('results'):
            print(f'SUCCESS: Found interaction between {drug1} and {drug2}!')
            interactions_text = data['results'][0].get('drug_interactions', [''])[0]
            print(interactions_text[:200])
    else:
        print(f"FAILED: {response.status_code}")

test_interaction("warfarin", "aspirin")
test_interaction("ibuprofen", "acetaminophen")

