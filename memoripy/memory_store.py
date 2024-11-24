import faiss
import numpy as np
import time
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from collections import defaultdict

from memoripy.interaction import Interaction

class MemoryStore:
    def __init__(self, dimension=1536):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.short_term_memory = []  # List of Interaction objects
        self.long_term_memory = []   # List of Interaction objects
        self.graph = nx.Graph()      # Graph for bidirectional associations
        self.semantic_memory = defaultdict(list)  # Semantic memory clusters
        self.core_memory = {
            "user_name": "User",
            "assistant_name": "Assistant",
            "persona": "helpful and professional",
        }

    def add_core_memory(self, interaction):
        """Add or update core memory attributes."""
        key = interaction.get("attribute_key")
        value = interaction.get("attribute_value")
        
        if key in ["user_name", "assistant_name", "persona"]:
            self.core_memory[key] = value
        else:
            print(f"Attribute key '{key}' is not supported for core memory.")

    def add_interaction(self, interaction_data):
        interaction_id = interaction_data['id']
        
        # Handle new message structure or fallback to old format
        if 'messages' in interaction_data:
            messages = interaction_data['messages']
        else:
            # Fallback for old format
            prompt = interaction_data.get('prompt', '')
            output = interaction_data.get('output', '')
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": output}]}
            ]

        # Create new Interaction object
        interaction = Interaction(
            id=interaction_id,
            messages=messages,
            embedding=np.array(interaction_data['embedding']).reshape(1, -1),
            timestamp=interaction_data.get('timestamp', time.time()),
            access_count=interaction_data.get('access_count', 1),
            concepts_list=set(interaction_data.get('concepts', [])),
            decay_factor=interaction_data.get('decay_factor', 1.0)
        )

        print(f"Adding new interaction to short-term memory: '{messages[0]['content'][0]['text']}'")

        # Save the interaction to short-term memory
        self.short_term_memory.append(interaction)
        self.index.add(interaction.embeddings)

        # Update graph with bidirectional associations
        self.update_graph(interaction.concepts_list)

        print(f"Total interactions stored in short-term memory: {len(self.short_term_memory)}")

    def update_graph(self, concepts):
        for concept in concepts:
            self.graph.add_node(concept)
        for concept1 in concepts:
            for concept2 in concepts:
                if concept1 != concept2:
                    if self.graph.has_edge(concept1, concept2):
                        self.graph[concept1][concept2]['weight'] += 1
                    else:
                        self.graph.add_edge(concept1, concept2, weight=1)

    def classify_memory(self):
        for interaction in self.short_term_memory[:]:
            if interaction.access_count > 10 and interaction not in self.long_term_memory:
                self.long_term_memory.append(interaction)
                print(f"Moved interaction {interaction.id} to long-term memory.")

    def retrieve(self, query_embedding, query_concepts, similarity_threshold=40, exclude_last_n=0):
        if len(self.short_term_memory) == 0:
            print("No interactions available in short-term memory for retrieval.")
            return []

        print("Retrieving relevant interactions from short-term memory...")
        relevant_interactions = []
        current_time = time.time()
        decay_rate = 0.0001

        # Normalize query embedding
        query_embedding_norm = normalize(query_embedding)

        # Calculate adjusted similarity for each interaction
        for idx, interaction in enumerate(self.short_term_memory[:-exclude_last_n if exclude_last_n else None]):
            # Normalize interaction embedding
            interaction_embedding_norm = normalize(interaction.embeddings)
            
            # Calculate similarity
            similarity = cosine_similarity(query_embedding_norm, interaction_embedding_norm)[0][0] * 100
            
            # Time-based decay
            time_diff = current_time - interaction.timestamp
            decay_factor = interaction.decay_factor * np.exp(-decay_rate * time_diff)
            
            # Reinforcement
            reinforcement_factor = np.log1p(interaction.access_count)
            
            # Adjusted similarity
            adjusted_similarity = similarity * decay_factor * reinforcement_factor
            print(f"Interaction {interaction.id} - Adjusted similarity score: {adjusted_similarity:.2f}%")

            if adjusted_similarity >= similarity_threshold:
                # Update interaction metrics
                interaction.access_count += 1
                interaction.timestamp = current_time
                interaction.decay_factor *= 1.1  # Increase decay factor for relevant interactions

                relevant_interactions.append((adjusted_similarity, interaction))
            else:
                # Decrease decay factor for non-relevant interactions
                interaction.decay_factor *= 0.9
                print(f"[DEBUG] Interaction {interaction.id} was not relevant (similarity: {adjusted_similarity:.2f}%)")

        # Handle spreading activation
        activated_concepts = self.spreading_activation(query_concepts)

        # Integrate spreading activation scores
        final_interactions = []
        for score, interaction in relevant_interactions:
            activation_score = sum([activated_concepts.get(c, 0) for c in interaction.concepts_list])
            total_score = score + activation_score
            final_interactions.append((total_score, interaction))

        # Sort and prepare final results
        final_interactions.sort(key=lambda x: x[0], reverse=True)
        final_interactions = [interaction for _, interaction in final_interactions]

        # Retrieve from semantic memory
        semantic_interactions = self.retrieve_from_semantic_memory(query_embedding_norm)
        final_interactions.extend(semantic_interactions)

        # Classify memories after retrieval
        self.classify_memory()

        print(f"Retrieved {len(final_interactions)} relevant interactions from memory.")
        return final_interactions

    def spreading_activation(self, query_concepts):
        print("Spreading activation for concept associations...")
        activated_nodes = {}
        initial_activation = 1.0
        decay_factor = 0.5

        for concept in query_concepts:
            activated_nodes[concept] = initial_activation

        for step in range(2):
            new_activated_nodes = {}
            for node in activated_nodes:
                if node in self.graph:
                    for neighbor in self.graph.neighbors(node):
                        if neighbor not in activated_nodes:
                            weight = self.graph[node][neighbor]['weight']
                            new_activation = activated_nodes[node] * decay_factor * weight
                            new_activated_nodes[neighbor] = new_activated_nodes.get(neighbor, 0) + new_activation
            activated_nodes.update(new_activated_nodes)

        print(f"Concepts activated after spreading: {activated_nodes}")
        return activated_nodes

    def cluster_interactions(self):
        print("Clustering interactions to create hierarchical memory...")
        if len(self.short_term_memory) < 2:
            print("Not enough interactions to perform clustering.")
            return

        embeddings_matrix = np.vstack([interaction.embeddings for interaction in self.short_term_memory])
        num_clusters = min(10, len(self.short_term_memory))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embeddings_matrix)
        
        # Clear existing semantic memory
        self.semantic_memory.clear()
        
        # Build semantic memory clusters
        for idx, label in enumerate(kmeans.labels_):
            self.semantic_memory[label].append(self.short_term_memory[idx])

        print(f"Clustering completed. Total clusters formed: {num_clusters}")

    def retrieve_from_semantic_memory(self, query_embedding_norm):
        print("Retrieving interactions from semantic memory...")
        if not self.semantic_memory:
            return []

        # Find the cluster closest to the query
        cluster_similarities = {}
        for label, interactions in self.semantic_memory.items():
            cluster_embeddings = np.vstack([interaction.embeddings for interaction in interactions])
            centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            centroid_norm = normalize(centroid)
            similarity = cosine_similarity(query_embedding_norm, centroid_norm)[0][0]
            cluster_similarities[label] = similarity

        best_cluster_label = max(cluster_similarities, key=cluster_similarities.get)
        print(f"Best matching cluster identified: {best_cluster_label}")

        # Retrieve and sort interactions from the best cluster
        cluster_interactions = self.semantic_memory[best_cluster_label]
        sorted_interactions = sorted(
            cluster_interactions,
            key=lambda x: cosine_similarity(query_embedding_norm, normalize(x.embeddings))[0][0],
            reverse=True
        )

        # Update access counts for retrieved interactions
        current_time = time.time()
        for interaction in sorted_interactions[:5]:
            interaction.access_count += 1
            interaction.timestamp = current_time
            print(f"[DEBUG] Updated access count for interaction {interaction.id}: {interaction.access_count}")

        print(f"Retrieved {len(sorted_interactions[:5])} interactions from the best matching cluster.")
        return sorted_interactions[:5]