import json
import os
import pandas as pd
import requests
class ETLTools:

    def extract_lode(self,url:str,output_folder:str,format:str):
        """
        This tool extracts the data from the API (url) and loads it into a 
        desired location(output_folder).

        Args:
            url (str): The API endpoint to extract data from.
            output_folder (str): The folder where the extracted data will be saved.
        
            Returns:
            str: The path to the saved data file.
         """

        project_root = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(project_root, output_folder)

        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()  # Assuming the API returns JSON data
            
            file_path = os.path.join(output_path, "extracted_data.json")
            os.makedirs(output_path, exist_ok=True)  # Create the output folder if it doesn't exist

            df = pd.json_normalize(data["results"])  # Convert the JSON data to a DataFrame
            if format == "csv":
                file_path = file_path.replace(".json", ".csv")
                df.to_csv(file_path, index=False)
            elif format == "json":
                df.to_json(file_path, orient="records", lines=True)
            else:
                raise ValueError("Unsupported format. Please choose 'csv' or 'json'.")

            return f"Data extracted and saved to {file_path}"
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while fetching data from {url}: {e}")
            raise

        return file_path

if __name__ == "__main__":
    obj = ETLTools()
    print(obj.extract_lode("https://pokeapi.co/api/v2/pokemon/", "data/extracted", "csv"))