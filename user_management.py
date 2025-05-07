import os
import json
import hashlib
import secrets
import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import shutil
from utils.helpers import make_serializable

USER_DB_PATH = "data/users.json"
USER_RESULTS_PATH = "data/user_results"

def initialize_database():
    """Initialize the user database and results directories if they don't exist"""
    # Create the data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Create the user results directory if it doesn't exist
    os.makedirs(USER_RESULTS_PATH, exist_ok=True)
    
    # Create the user database file if it doesn't exist
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, 'w') as f:
            json.dump({}, f)
        print(f"Created user database at {USER_DB_PATH}")

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    """
    Hash a password with a salt for secure storage
    
    Args:
        password: The plaintext password to hash
        salt: Optional salt to use, if None a new salt is generated
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Create a hash of the password with the salt
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # Number of iterations
    ).hex()
    
    return password_hash, salt

def get_user_db() -> Dict:
    """Load the user database from disk"""
    if not os.path.exists(USER_DB_PATH):
        initialize_database()
    
    with open(USER_DB_PATH, 'r') as f:
        return json.load(f)

def save_user_db(user_db: Dict):
    """Save the user database to disk"""
    with open(USER_DB_PATH, 'w') as f:
        json.dump(user_db, f, indent=4)

def user_exists(username: str) -> bool:
    """Check if a user exists in the database"""
    user_db = get_user_db()
    return username in user_db

def register_user(username: str, password: str) -> bool:
    """
    Register a new user
    
    Args:
        username: The username for the new account
        password: The password for the new account
        
    Returns:
        True if registration was successful, False otherwise
    """
    # Check if the username already exists
    if user_exists(username):
        return False
    
    # Hash the password with a new salt
    password_hash, salt = hash_password(password)
    
    # Get the user database
    user_db = get_user_db()
    
    # Create the user record
    user_db[username] = {
        "password_hash": password_hash,
        "salt": salt,
        "created_at": datetime.datetime.now().isoformat(),
        "last_login": None
    }
    
    # Create user's results directory
    os.makedirs(os.path.join(USER_RESULTS_PATH, username), exist_ok=True)
    
    # Save the updated database
    save_user_db(user_db)
    
    return True

def authenticate_user(username: str, password: str) -> bool:
    """
    Authenticate a user with their username and password
    
    Args:
        username: The username to authenticate
        password: The password to verify
        
    Returns:
        True if authentication was successful, False otherwise
    """
    # Get the user database
    user_db = get_user_db()
    
    # Check if the username exists
    if username not in user_db:
        return False
    
    # Get the user record
    user = user_db[username]
    
    # Hash the provided password with the stored salt
    password_hash, _ = hash_password(password, user["salt"])
    
    # Check if the password hash matches
    if password_hash == user["password_hash"]:
        # Update last login time
        user_db[username]["last_login"] = datetime.datetime.now().isoformat()
        save_user_db(user_db)
        return True
    
    return False

def save_user_result(username: str, result_data: Dict, image_paths: Dict = None) -> str:
    """
    Save a user's fruit analysis result
    
    Args:
        username: The username of the user
        result_data: Dictionary containing the analysis results
        image_paths: Dictionary of image paths to save with the result
        
    Returns:
        ID of the saved result
    """
    # Create a unique ID for this result
    result_id = f"{username}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Create a directory for this result
    result_dir = os.path.join(USER_RESULTS_PATH, username, result_id)
    os.makedirs(result_dir, exist_ok=True)
    
    # Create a results metadata file
    metadata = {
        "id": result_id,
        "username": username,
        "timestamp": datetime.datetime.now().isoformat(),
        "fruit_type": result_data.get("fruit_type", "Unknown"),
        "analysis_type": "multi_angle" if result_data.get("multi_angle", False) else "single_image",
        "ripeness_predictions": result_data.get("ripeness_predictions", []),
    }
    
    # Save the metadata
    with open(os.path.join(result_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
    
    # Save the full result data
    with open(os.path.join(result_dir, "full_result.json"), 'w') as f:
        # Remove image objects from the data to avoid serialization issues
        result_data_copy = result_data.copy()
        if "original_image" in result_data_copy:
            del result_data_copy["original_image"]
        if "segmented_image" in result_data_copy:
            del result_data_copy["segmented_image"]
        if "mask" in result_data_copy:
            result_data_copy["mask"] = result_data_copy["mask"].tolist() if hasattr(result_data_copy["mask"], "tolist") else None
            
        # Convert result_data_copy to serializable format
        result_data_serializable = convert_to_serializable(result_data_copy)
        
        json.dump(result_data_serializable, f, indent=4)
    
    # Save the images if provided
    if image_paths and isinstance(image_paths, dict):
        for image_type, image_path in image_paths.items():
            if os.path.isfile(image_path):
                # Copy the image to the result directory
                import shutil
                shutil.copy2(
                    image_path, 
                    os.path.join(result_dir, f"{image_type}.png")
                )
    
    return result_id

def save_enhanced_user_result(username, result_data, image_paths=None):
    """
    Save a user's enhanced two-stage fruit analysis result with improved error handling
    
    Args:
        username: The username of the user
        result_data: Dictionary containing the enhanced analysis results
        image_paths: Dictionary of image paths to save with the result
        
    Returns:
        ID of the saved result
    """
    import os
    import json
    import datetime
    import shutil
    import traceback
    
    try:
        # Create a unique ID for this result
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        result_id = f"{username}_enhanced_{timestamp}"
        
        # Create a directory for this result
        results_path = "data/user_results"
        result_dir = os.path.join(results_path, username, result_id)
        os.makedirs(result_dir, exist_ok=True)
        
        # Determine fruit type and number of fruits
        fruit_type = result_data.get("fruit_type", "Unknown")
        num_fruits = result_data.get("num_fruits", 1)
        
        # Create metadata for this analysis
        metadata = {
            "id": result_id,
            "username": username,
            "timestamp": datetime.datetime.now().isoformat(),
            "fruit_type": fruit_type,
            "analysis_type": "enhanced_two_stage",
            "num_fruits": num_fruits,
            "user_note": result_data.get("user_note", "")
        }
        
        # Find highest confidence ripeness if available
        if "confidence_distributions" in result_data and result_data["confidence_distributions"]:
            first_dist = result_data["confidence_distributions"][0]
            if first_dist and isinstance(first_dist, dict):
                # Filter out non-confidence keys
                filtered_dist = {k: v for k, v in first_dist.items() 
                               if k not in ["error", "estimated"] and isinstance(v, (int, float))}
                
                # Find highest confidence ripeness
                if filtered_dist:
                    highest_ripeness = max(filtered_dist.items(), key=lambda x: x[1])
                    metadata["primary_ripeness"] = highest_ripeness[0]
                    metadata["primary_confidence"] = highest_ripeness[1]
        
        # Save the metadata
        metadata_path = os.path.join(result_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Save the serialized result data
        result_data_path = os.path.join(result_dir, "full_result.json")
        with open(result_data_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        # Save image files if provided
        if image_paths and isinstance(image_paths, dict):
            for image_type, image_path in image_paths.items():
                if os.path.isfile(image_path):
                    # Create a descriptive filename
                    if image_type == "original":
                        output_filename = f"original_{fruit_type}.png"
                    elif image_type == "segmented":
                        output_filename = f"segmented_{fruit_type}.png"
                    elif image_type == "mask":
                        output_filename = f"mask_{fruit_type}.png"
                    elif image_type == "bounding_box_visualization":
                        output_filename = f"detection_{fruit_type}.png"
                    else:
                        # Default case
                        output_filename = f"{image_type}.png"
                    
                    # Copy the file
                    try:
                        dst_path = os.path.join(result_dir, output_filename)
                        shutil.copy2(image_path, dst_path)
                    except Exception as e:
                        print(f"Warning: Failed to copy {image_path} to {dst_path}: {str(e)}")
        
        return result_id
        
    except Exception as e:
        print(f"Error saving enhanced user result: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Failed to save results: {str(e)}")

def convert_to_serializable(obj):
    """
    Convert a data structure with NumPy values to JSON-serializable types
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON-serializable version of the object
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def get_user_results(username: str) -> List[Dict]:
    """
    Get all results for a user
    
    Args:
        username: The username to get results for
        
    Returns:
        List of result metadata dictionaries
    """
    results = []
    
    # Get the user's results directory
    user_results_dir = os.path.join(USER_RESULTS_PATH, username)
    
    # Check if the directory exists
    if not os.path.exists(user_results_dir):
        return results
    
    # Get all result directories for this user
    for result_id in os.listdir(user_results_dir):
        result_dir = os.path.join(user_results_dir, result_id)
        
        # Check if this is a directory
        if not os.path.isdir(result_dir):
            continue
        
        # Check if there's a metadata file
        metadata_path = os.path.join(result_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            continue
        
        # Load the metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Add image paths
        metadata["image_paths"] = {}
        for image_type in ["original", "segmented", "combined_visualization", "bounding_box_visualization", "comparison_visualization"]:
            img_path = os.path.join(result_dir, f"{image_type}.png")
            if os.path.exists(img_path):
                metadata["image_paths"][image_type] = img_path
        
        results.append(metadata)
    
    # Sort by timestamp (newest first)
    results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return results

def get_result_details(username: str, result_id: str) -> Optional[Dict]:
    """
    Get detailed information about a specific result
    
    Args:
        username: The username of the result owner
        result_id: The ID of the result to retrieve
        
    Returns:
        Dictionary containing the result details or None if not found
    """
    # Check if the result exists
    result_dir = os.path.join(USER_RESULTS_PATH, username, result_id)
    if not os.path.exists(result_dir):
        return None
    
    # Get the full result data
    full_result_path = os.path.join(result_dir, "full_result.json")
    if not os.path.exists(full_result_path):
        return None
    
    with open(full_result_path, 'r') as f:
        result_data = json.load(f)
    
    # Get the image paths
    image_paths = {}
    for image_type in ["original", "segmented", "combined_visualization", "bounding_box_visualization", "comparison_visualization"]:
        img_path = os.path.join(result_dir, f"{image_type}.png")
        if os.path.exists(img_path):
            image_paths[image_type] = img_path
    
    result_data["image_paths"] = image_paths
    
    return result_data

def delete_user_result(username: str, result_id: str) -> bool:
    """
    Delete a specific analysis result for a user
    """
    
    # Get the result directory path
    result_dir = os.path.join(USER_RESULTS_PATH, username, result_id)
    
    # Check if the directory exists
    if not os.path.exists(result_dir):
        return False
    
    try:
        shutil.rmtree(result_dir)
        return True
    except Exception as e:
        print(f"Error deleting result {result_id}: {e}")
        return False