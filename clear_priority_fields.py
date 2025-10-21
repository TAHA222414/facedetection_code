import firebase_admin
from firebase_admin import credentials, firestore

# 1) Point to your service account JSON (same one you use in face.py)
cred = credentials.Certificate("serviceAccountKey.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

def clear_priority(collection_name: str):
    print(f"🔹 Scanning collection: {collection_name}")
    docs = db.collection(collection_name).stream()
    changed = 0
    for doc in docs:
        data = doc.to_dict() or {}
        if 'priority' in data:
            db.collection(collection_name).document(doc.id).update({
                'priority': firestore.DELETE_FIELD
            })
            changed += 1
            print(f"   • Removed priority from doc: {doc.id}")
    print(f"✅ Finished '{collection_name}': {changed} document(s) updated.\n")

if __name__ == "__main__":
    clear_priority("notices")
    clear_priority("notices_fr")
