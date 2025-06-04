import { Component } from '@angular/core'; // Imports the base Angular component class for building components
import { HttpClient } from '@angular/common/http'; // Imports HttpClient to make HTTP requests
 // Provides common Angular directives such as ngIf and ngFor
import { FormsModule } from '@angular/forms'; // Provides forms-related functionality such as two-way data binding
import { HttpClientModule } from '@angular/common/http'; // Imports HttpClientModule for making HTTP requests (module version of HttpClient)

@Component({
  selector: 'app-production-settings',
   imports: [
    FormsModule,
    HttpClientModule
],
  templateUrl: './production-settings.component.html',
  styleUrl: './production-settings.component.css'
})
export class ProductionSettingsComponent {
  title = 'Frontend'; // Defines the title of the app
  dbNames: string[] = []; // Array to store the names of the datasets
  modelNames: string[] = []; // Array to store the names of the models

  graphicsNamesOne: string[] = []; // Array to store the graphic names for SHAP Feature 1
  selectedGraphicOne: string = ''; // Variable to hold the selected graphic for SHAP Feature 1
  nameDBOne: string = ''; // Variable to hold the selected database name for SHAP Feature 1
  nameModelOne: string = ''; // Variable to hold the selected model name for SHAP Feature 1
  graphicURLOne: string = ''; // Variable to store the URL of the selected graphic for SHAP Feature 1

  graphicsNamesEight: string[] = []; // Array to store the graphic names for SHAP Feature 8
  selectedGraphicEight: string = ''; // Variable to hold the selected graphic for SHAP Feature 8
  nameDBEight: string = ''; // Variable to hold the selected database name for SHAP Feature 8
  nameModelEight: string = ''; // Variable to hold the selected model name for SHAP Feature 8
  graphicURLEight: string = ''; // Variable to store the URL of the selected graphic for SHAP Feature 8
  nameDBLime: string = '';
  nameModelLime: string = ''; 
  InstanceOne: number = 0; // ID da instância para SHAP One
  InstanceEight: number = 0; // ID da instância para SHAP Eight
  InstanceLime: number = 0; // ID da instância para LIME

  constructor(private http: HttpClient) {} // Injects HttpClient into the constructor for making HTTP requests

  /* Dataset */
  fetchDbNames() {
    // Fetches the list of dataset files from the backend
    this.http.get<{ files: string[] }>('http://127.0.0.2:5000/api_dataset/files_dataset').subscribe({
      next: (data) => {
        // Updates dbNames by adding new files not already in the list
        const newFiles = Array.isArray(data.files) ? data.files : [];
        this.dbNames = [...this.dbNames, ...newFiles.filter(file => !this.dbNames.includes(file))];
      },
      error: (err) => console.error('Error fetching DB names:', err), // Logs error if the request fails
    });
  }

  /* Model */
  fetchModelNames() {
    // Fetches the list of model files from the backend
    this.http.get<{ files: string[] }>('http://127.0.0.3:5000/api_model/files_model').subscribe({
      next: (data) => {
        // Updates modelNames by adding new files not already in the list
        const newFiles = Array.isArray(data.files) ? data.files : [];
        this.modelNames = [...this.modelNames, ...newFiles.filter(file => !this.modelNames.includes(file))];
      },
      error: (err) => console.error('Error fetching model names:', err), // Logs error if the request fails
    });
  }

  // Sets the database name for SHAP Feature 1 and SHAP Feature 8
  setDbName(db: string) {
    this.nameDBOne = db; // Assigns the selected database name for SHAP Feature 1
    this.nameDBEight = db; // Assigns the selected database name for SHAP Feature 8
    this.nameDBLime = db;
  }

  // Sets the model name for SHAP Feature 1 and SHAP Feature 8
  setModelName(model: string) {
    this.nameModelOne = model; // Assigns the selected model name for SHAP Feature 1
    this.nameModelEight = model; // Assigns the selected model name for SHAP Feature 8
    this.nameModelLime = model; 
  }

  /* Shap Feature 1 */
  sendDataRequestOne() {
    // Sends a request to load data for SHAP Feature 1
    const url = `http://127.0.0.4:5000/api_shap_1/send_data_request/${this.nameDBOne}/${this.nameModelOne}`;
    this.http.get(url).subscribe({
      next: () => alert('Model loading and dataset successfully!'), // Alerts success message
      error: (err) => console.error('Error sending request:', err), // Logs error if the request fails
    });
  }

  generateGraphicsOne() {
    // Sends a request to generate graphics for SHAP Feature 1
    const url = `http://127.0.0.4:5000/api_shap_1/generation_graphics`;
    this.http.get(url).subscribe({
      next: () => alert('Graphics shap generated successfully!'), // Alerts success message
      error: (err) => console.error('Error generating graphics:', err), // Logs error if the request fails
    });
  }

  /* Shap Feature 8 */
  sendDataRequestEight() {
    // Sends a request to load data for SHAP Feature 8
    const url = `http://127.0.0.5:5000/api_shap_8/send_data_request/${this.nameDBEight}/${this.nameModelEight}`;
    this.http.get(url).subscribe({
      next: () => alert('Model loading and dataset successfully!'), // Alerts success message
      error: (err) => console.error('Error sending request:', err), // Logs error if the request fails
    });
  }

  generateGraphicsEight() {
    // Sends a request to generate graphics for SHAP Feature 8
    const url = `http://127.0.0.5:5000/api_shap_8/generation_graphics`;
    this.http.get(url).subscribe({
      next: () => alert('Graphics shap generated successfully!'), // Alerts success message
      error: (err) => console.error('Error generating graphics:', err), // Logs error if the request fails
    });
  }

    /* Lime */
    sendDataRequestLime() {
      const url = `http://127.0.0.6:5000/api_lime/send_data_request/${this.nameDBLime}/${this.nameModelLime}`;
      this.http.get(url).subscribe({
        next: () => alert('Model loading and dataset successfully!'), // Alerts success message
        error: (err) => console.error('Error sending request:', err), // Logs error if the request fails
      });
    }
  
    generateGraphicsLime() {
  
      const url = `http://127.0.0.6:5000/api_lime/generation_graphics`;
      this.http.get(url).subscribe({
        next: () => alert('Graphics lime generated successfully!'), // Alerts success message
        error: (err) => console.error('Error generating graphics:', err), // Logs error if the request fails
      });
    }

    /* SHAP One */
  performShapOneCalculation() {
    const url = `http://127.0.0.4:5000/api_shap_1/perform_shap_calculations/${this.InstanceOne}`;
    this.http.get(url).subscribe({
      next: () => alert('SHAP One calculations completed successfully!'),
      error: (err) => console.error('Error performing SHAP One calculations:', err),
    });
  }

  /* SHAP Eight */
  performShapEightCalculation() {
    const url = `http://127.0.0.5:5000/api_shap_8/perform_shap_calculations/${this.InstanceEight}`;
    this.http.get(url).subscribe({
      next: () => alert('SHAP Eight calculations completed successfully!'),
      error: (err) => console.error('Error performing SHAP Eight calculations:', err),
    });
  }

  /* LIME */
  performLimeCalculation() {
    const url = `http://127.0.0.6:5000/api_lime/perform_lime_calculations/${this.InstanceLime}`;
    this.http.get(url).subscribe({
      next: () => alert('LIME calculations completed successfully!'),
      error: (err) => console.error('Error performing LIME calculations:', err),
    });
  }
}