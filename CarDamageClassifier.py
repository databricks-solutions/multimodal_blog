import mlflow.pyfunc
from mlflow.models import set_model
import pandas as pd
import numpy as np
from PIL import Image
import io
import json
from typing import List, Literal, Any, Dict, Optional
from pydantic import BaseModel
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import base64
from copy import deepcopy

# Pydantic models for damage classification
class DamageType(BaseModel):
    part: Literal[
        "Front Bumper",
        "Rear Bumper",
        "Fender",
        "Hood",
        "Trunk Lid",
        "Roof Panel",
        "Door Panel",
        "Quarter Panel",
        "Windshield",
        "Side Window",
        "Rear Window",
        "Headlamp",
        "Tail Lamp",
        "Unknown"
    ]

class DamageAnalysis(BaseModel):
    is_damaged: bool
    damage_list: List[DamageType]


class CarDamageClassifier(mlflow.pyfunc.PythonModel):
    
    def __init__(self):
        """
        Initialize the model. Clients will be initialized in load_context.
        """
        self.deploy_client = None
        self.vector_search_client = None
        self.vector_index = None
        
    def load_context(self, context):
        """
        Load the model context and initialize clients.
        This is called once when the model is loaded.
        """
        import mlflow.deployments
        from databricks.vector_search.client import VectorSearchClient
        
        # Initialize deployment client for accessing endpoints
        # This will use the environment's authentication automatically
        self.deploy_client = mlflow.deployments.get_deploy_client("databricks")
        
        # Initialize Vector Search Client
        # This will also use the environment's authentication
        self.vector_search_client = VectorSearchClient()
        
        # Get the vector search index
        try:
            self.vector_index = self.vector_search_client.get_index(
                endpoint_name="multi_modal_blog_endpoint",
                index_name="users.colton_peltier.classified_damages_gold_index"
            )
        except Exception as e:
            print(f"Warning: Could not initialize vector index: {e}")
            self.vector_index = None
    
    @mlflow.trace(name="car_damage_classification_process_image", span_type="PARSER")
    def resize_single_image(self, content: bytes) -> bytes:
        """Resize a single image to max dimension of 336px while maintaining aspect ratio"""
        buffer = io.BytesIO()
        max_side = 336
        img = Image.open(io.BytesIO(content))
        img.thumbnail((max_side, max_side))
        img.save(buffer, format="JPEG")
        return buffer.getvalue()
    
    def recursive_schema_flatten(self, schema: Any, refs_dict: dict) -> dict:
        """Recursively flatten schema references"""
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref = schema["$ref"]
                reference_name = ref.split("/")[-1]
                return self.recursive_schema_flatten(refs_dict[reference_name], refs_dict)
            return {k: self.recursive_schema_flatten(v, refs_dict) for k, v in schema.items()}
        if isinstance(schema, list):
            return [self.recursive_schema_flatten(v, refs_dict) for v in schema]
        return schema

    def pydantic_to_ai_query_json_schema(self, pydantic_model: BaseModel, name: str, strict: bool) -> dict:
        """Convert Pydantic model to flattened JSON schema for AI query"""
        _model = deepcopy(pydantic_model)
        model_schema = _model.model_json_schema()
        refs = model_schema.get("$defs", {})
        # Delete the refs from the schema
        model_schema.pop("$defs", None)
        return {
            "name": name,
            "schema": self.recursive_schema_flatten(model_schema, refs),
            "strict": strict
        }
    
    @mlflow.trace(name="car_damage_classification_create_prompt", span_type="PARSER")
    def _create_prompt(self, image_base64: str) -> dict:
        """Create the prompt for Llama 4 Maverick with structured output requirements"""
        
        # Generate the flattened schema for the response format
        response_schema = self.pydantic_to_ai_query_json_schema(
            DamageAnalysis,
            name="damage_analysis",
            strict=True
        )
        
        # Create a human-readable description of the schema for the system prompt
        schema_description = json.dumps(response_schema["schema"], indent=2)
        
        system_prompt = f"""You are an expert mechanic working at a body repair shop. Your job is to look at an image and accurately list any body panels which are damaged on the car in the image. If multiple of the same part are damaged (like 2 doors) list that part mulitple times. If the part is unknown, choose Unknown.\n\nYou must respond with a JSON object that strictly follows this schema: {schema_description}"""
        
        return {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please analyze this car image for damage and provide a structured assessment following the specified JSON schema."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.1,  # Low temperature for consistent classification
            "response_format": {
                "type": "json_schema",
                "json_schema": response_schema
            }
        }
    
    @mlflow.trace(name="car_damage_classification_call_llm", span_type="CHAT_MODEL")
    def _call_llama_api(self, image_bytes: bytes) -> DamageAnalysis:
        """Call Databricks-hosted Llama 4 Maverick API via Foundation Models"""
        
        try:
            # Convert image to base64 for API transmission
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create the request payload using the same prompt structure
            payload = self._create_prompt(image_base64)
            
            # Call the model using MLflow deployments client
            response = self.deploy_client.predict(
                endpoint="databricks-llama-4-maverick",
                inputs=payload
            )
            
            # Extract the JSON from the model's response
            if isinstance(response, dict):
                if "choices" in response:
                    content = response["choices"][0]["message"]["content"]
                else:
                    content = response.get("predictions", [{}])[0].get("content", "{}")
            else:
                content = str(response)
            
            # Parse the JSON response into our Pydantic model
            damage_data = json.loads(content)
            
            # Validate with Pydantic
            analysis = DamageAnalysis(**damage_data)
            
            return analysis
            
        except Exception as e:
            print(f"Error in _call_llama_api: {type(e).__name__}: {str(e)}")
            return DamageAnalysis(is_damaged=False, damage_list=[])
    
    @mlflow.trace(name="car_damage_classification_get_embeddings", span_type="EMBEDDING")
    def _get_damage_embeddings(self, damaged_parts: List[str]) -> np.ndarray:
        """
        Get embeddings for the damaged parts list using the embeddings endpoint.
        Sums embeddings for multiple damage parts.
        """
        # Check if damaged_parts is empty or None
        if not damaged_parts:
            return np.zeros(1024)
        
        try:
            embeddings_list = []
            
            # Get embeddings for each damage part and sum them
            for part in damaged_parts:
                try:
                    # Use the deployments client to get embeddings
                    response = self.deploy_client.predict(
                        endpoint="databricks-gte-large-en",
                        inputs={"input": part}
                    )
                    
                    # Extract embedding from response
                    if isinstance(response, dict):
                        if "data" in response and response["data"]:
                            embedding = np.array(response["data"][0]["embedding"])
                            embeddings_list.append(embedding)
                        elif "embeddings" in response:
                            embedding = np.array(response["embeddings"][0])
                            embeddings_list.append(embedding)
                            
                except Exception as e:
                    print(f"Error getting embedding for part '{part}': {type(e).__name__}: {str(e)}")
            
            # Sum all embeddings
            if embeddings_list:
                return_embeddings = np.sum(embeddings_list, axis=0)
                return return_embeddings
            else:
                # Return zero vector if no embeddings were successfully retrieved
                return np.zeros(1024)
                
        except Exception as e:
            print(f"Error in _get_damage_embeddings: {type(e).__name__}: {str(e)}")
            # Return zero vector on any fatal error
            return np.zeros(1024)
    
    @mlflow.trace(name="car_damage_classification_vector_search", span_type="RETRIEVER")
    def _find_similar_claims(self, damaged_parts: List[str]) -> dict:
        """
        Find similar claims using vector search and calculate estimated quote.
        """
        if not self.vector_index:
            print("Vector index not available, skipping similarity search")
            return {
                'estimated_quote': 0.0,
                'similar_claim_ids': [],
                'similar_claims_count': 0,
                'similar_claims_details': []
            }
            
        try:
            # Get embeddings for the damaged parts
            damage_embeddings = self._get_damage_embeddings(damaged_parts)
            
            # Perform similarity search using the loaded vector index
            results = self.vector_index.similarity_search(
                query_vector=damage_embeddings.tolist(),
                columns=["damage_list", "claim_id", "final_cost_to_customer"],
                num_results=5
            )
            
            # Extract similar claims data
            similar_claims = []
            total_cost = 0.0
            claim_ids = []
            
            if results and 'result' in results and 'data_array' in results['result']:
                for claim_data in results['result']['data_array']:
                    damage_list, claim_id, cost, score = claim_data
                    similar_claims.append({
                        'damage_list': damage_list,
                        'claim_id': claim_id,
                        'cost': cost,
                        'similarity_score': score
                    })
                    total_cost += cost
                    claim_ids.append(claim_id)
                
                # Calculate average cost
                avg_cost = total_cost / len(similar_claims) if similar_claims else 0.0
                
                return {
                    'estimated_quote': round(avg_cost, 2),
                    'similar_claim_ids': claim_ids,
                    'similar_claims_count': len(similar_claims),
                    'similar_claims_details': similar_claims
                }
            else:
                return {
                    'estimated_quote': 0.0,
                    'similar_claim_ids': [],
                    'similar_claims_count': 0,
                    'similar_claims_details': []
                }
                
        except Exception as e:
            print(f"Error in vector search: {str(e)}")
            return {
                'estimated_quote': 0.0,
                'similar_claim_ids': [],
                'similar_claims_count': 0,
                'similar_claims_details': [],
                'error': str(e)
            }
    
    @mlflow.trace(name="car_damage_classification_predict", span_type="AGENT")
    def predict(self, context, model_input, params=None):
        """
        Process car images for damage detection and estimate repair costs.
        
        Args:
            context: MLflow PythonModelContext (contains artifacts path, etc.)
            model_input: pandas DataFrame with 'image' column containing image bytes
            params: Optional parameters dictionary
            
        Returns:
            pandas DataFrame with damage classification and cost estimation results
        """
        results = []
        
        # Process each image
        for idx, row in model_input.iterrows():
            try:
                # Get image bytes from the row
                image_bytes = row['image']
                
                # Resize the image
                resized_image = self.resize_single_image(image_bytes)
                
                # Call Llama API for classification
                analysis = self._call_llama_api(resized_image)
                
                # Format results for output
                damage_parts = [damage.part for damage in analysis.damage_list]
                
                # Initialize result dictionary
                result = {
                    'is_damaged': analysis.is_damaged,
                    'damage_count': len(analysis.damage_list),
                    'damaged_parts': json.dumps(damage_parts),  # Convert to JSON string
                    'damage_details': analysis.model_dump_json(),
                    'estimated_quote': 0.0,
                    'similar_claim_ids': json.dumps([]),  # Convert to JSON string
                    'similar_claims_count': 0
                }
                
                # If damage is detected, find similar claims and estimate cost
                if analysis.is_damaged and damage_parts:
                    similarity_results = self._find_similar_claims(damage_parts)
                    result.update({
                        'estimated_quote': similarity_results['estimated_quote'],
                        'similar_claim_ids': json.dumps(similarity_results['similar_claim_ids']),
                        'similar_claims_count': similarity_results['similar_claims_count']
                    })
                
                results.append(result)
                
            except Exception as e:
                # Handle any unexpected errors
                print(f"Error processing image at index {idx}: {str(e)}")
                results.append({
                    'is_damaged': False,
                    'damage_count': 0,
                    'damaged_parts': json.dumps([]),
                    'damage_details': json.dumps({"error": str(e)}),
                    'estimated_quote': 0.0,
                    'similar_claim_ids': json.dumps([]),
                    'similar_claims_count': 0
                })
        
        # Convert results to DataFrame to match the output schema
        return pd.DataFrame(results)


# Set the model for serving
set_model(CarDamageClassifier())
