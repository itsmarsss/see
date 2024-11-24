# json_storage.py

import json

class JSONStorage():
    def __init__(self, file_path="interaction_history.json"):
        self.file_path = file_path

    def load_history(self):
        try:
            with open(self.file_path, 'r') as f:
                history = json.load(f)
            return (
                history.get("short_term_memory", []),
                history.get("long_term_memory", []),
                history.get("core_memory", [])
            )
        except FileNotFoundError:
            return [], [], []

    def save_memory_to_history(self, memory_store):
        history = {
            "short_term_memory": [],
            "long_term_memory": [],
            "core_memory": []
        }

        # Save short-term memory interactions with all data
        for idx in range(len(memory_store.short_term_memory)):
            history_entry = {
                'id': memory_store.short_term_memory[idx].id,
                'messages': memory_store.short_term_memory[idx].messages,
                'embedding': memory_store.short_term_memory[idx].embeddings.flatten().tolist(),
                'timestamp': memory_store.short_term_memory[idx].timestamp,
                'access_count': memory_store.short_term_memory[idx].access_count,
                'concepts': list(memory_store.short_term_memory[idx].concepts_list),
                'decay_factor': memory_store.short_term_memory[idx].decay_factor
            }
            history["short_term_memory"].append(history_entry)

        # Save long-term memory interactions with all data
        for idx in range(len(memory_store.long_term_memory)):
            history_entry = {
                'id': memory_store.long_term_memory[idx].id,
                'messages': memory_store.long_term_memory[idx].messages,
                'embedding': memory_store.long_term_memory[idx].embeddings.flatten().tolist(),
                'timestamp': memory_store.long_term_memory[idx].timestamp,
                'access_count': memory_store.long_term_memory[idx].access_count,
                'concepts': list(memory_store.long_term_memory[idx].concepts),
                'decay_factor': memory_store.long_term_memory[idx].decay_factor
            }
            history["long_term_memory"].append(history_entry)

        # Save core memory with complete data
        #for idx in range(len(memory_store.core_embeddings)):
        #    history_entry = {
        #        'embedding': memory_store.core_embeddings[idx].flatten().tolist(),
        #        'timestamp': memory_store.core_timestamps[idx],
        #        'concepts': list(memory_store.core_concepts[idx]),
        #        'attribute_key': memory_store.core_memory.get('attribute_key', ''),
        #        'attribute_value': memory_store.core_memory.get('attribute_value', ''),
        #    }
        #    history["core_memory"].append(history_entry)

        # Also save current core memory state
        history["core_memory_state"] = memory_store.core_memory

        with open(self.file_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Saved interaction history to JSON. Short-term: {len(history['short_term_memory'])}, Long-term: {len(history['long_term_memory'])}")
