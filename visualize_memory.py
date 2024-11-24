import json
from pyvis.network import Network
from collections import defaultdict
import os

def visualize_memory_graph_pyvis(json_file="interaction_history.json"):
    try:
        # Load memory data
        with open(json_file) as f:
            data = json.load(f)
        
        # Create network with notebook=False
        net = Network(
            height="100vh",  # Changed to viewport height
            width="100%",
            bgcolor="#222222",  # Dark background
            font_color="white",  # Light text
            notebook=False
        )
        
        # Configure physics
        net.force_atlas_2based()
        net.show_buttons(filter_=['physics'])  # Uncommented this line
        net.barnes_hut()

        # Track frequencies and edges
        concept_freq = defaultdict(int)
        edge_weights = defaultdict(int)
        
        # Build connections
        for interaction in data.get("short_term_memory", []):
            concepts = interaction.concepts
            for concept in concepts:
                concept_freq[concept] += 1
                
            # Connect concepts
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i+1:]:
                    edge_weights[(c1, c2)] += 1
        
        # Add nodes
        max_freq = max(concept_freq.values()) if concept_freq else 1
        for concept, freq in concept_freq.items():
            size = 20 + (freq / max_freq) * 30
            net.add_node(
                concept,
                label=concept,
                title=f"Frequency: {freq}",
                size=size,
                color={'background': '#2b5876', 'border': '#4e4376'}  # Dark blue gradient
            )
        
        # Add edges
        max_weight = max(edge_weights.values()) if edge_weights else 1
        for (c1, c2), weight in edge_weights.items():
            width = 1 + (weight / max_weight) * 5
            net.add_edge(
                c1, c2,
                width=width,
                title=f"Co-occurrences: {weight}",
                color={'color': '#6e8898'}  # Muted blue-gray for edges
            )
        
        # Generate HTML content
        output_path = os.path.abspath('memory_graph.html')
        html_content = net.generate_html(name=output_path, local=True, notebook=False)
        
        # Add our CSS directly to the HTML content
        css = """
        <style>
        html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; }
        body { 
            background: #222222;
        }
        #mynetwork { 
            width: 100vw !important; 
            height: 100vh !important; 
            background: #222222 !important;
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
        }
        .vis-configuration-wrapper {
            position: fixed !important;
            top: 0 !important;
            right: -530px !important;  /* Start offscreen */
            height: 100vh !important;
            background: rgba(34, 34, 34, 0.6) !important;  /* More transparent */
            backdrop-filter: blur(8px) !important;
            -webkit-backdrop-filter: blur(8px) !important;  /* For Safari */
            padding: 20px !important;
            z-index: 1000 !important;
            color: white !important;
            width: 530 !important;
            transition: right 0.5s ease-out !important;
        }
        #menu-toggle {
            position: absolute !important;
            top: 50% !important;
            left: 0 !important;
            transform: translate(-100%, -50%) !important;
            background: rgba(34, 34, 34, 0.95) !important;
            color: white !important;
            border: none !important;
            padding: 8px 3px !important;
            cursor: pointer !important;
            z-index: 1001 !important;
            border-radius: 4px 0 0 4px !important;
            font-size: 10px !important;
        }
        .menu-visible .vis-configuration-wrapper {
            right: 0 !important;
        }
        #menu-toggle::after {
            content: '◀' !important;
            display: block !important;
            transform: scaleY(2) !important;
        }
        .menu-visible #menu-toggle::after {
            content: '▶' !important;
        }
        .vis-configuration {
            background: transparent !important;
            color: white !important;
        }
        .vis-config-item {
            color: white !important;
        }
        .vis-configuration input {
            background: rgba(51, 51, 51, 0.8) !important;
            color: white !important;
            border: 1px solid rgba(68, 68, 68, 0.8) !important;
        }
        .vis-configuration select {
            background: rgba(51, 51, 51, 0.8) !important;
            color: white !important;
        }
        </style>
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            const wrapper = document.querySelector('.vis-configuration-wrapper');
            const toggle = document.createElement('button');
            toggle.id = 'menu-toggle';
            wrapper.appendChild(toggle);
            
            toggle.addEventListener('click', function() {
                document.body.classList.toggle('menu-visible');
            });
        });
        </script>"""
        
        # Inject CSS only - remove the previous container wrapper modification
        modified_html = html_content.replace('<head>', f'<head>{css}')
        
        # Write the modified HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modified_html)
            
        print(f"Graph saved to: {output_path}")
        os.system(f"start {output_path}")
        
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")

if __name__ == "__main__":
    visualize_memory_graph_pyvis()