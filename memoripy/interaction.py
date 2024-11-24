class Interaction:
    def __init__(self, id, messages, embedding, timestamp, access_count, concepts_list, decay_factor):
        self.id = id
        self.messages = messages
        self.embeddings = embedding
        self.timestamp = timestamp
        self.access_count = access_count
        self.concepts_list = concepts_list
        self.decay_factor = decay_factor