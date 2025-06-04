import { Component } from '@angular/core'; // Imports the base Angular component class for building components
import { HttpClient } from '@angular/common/http'; // Imports HttpClient to make HTTP requests
 // Provides common Angular directives such as ngIf and ngFor
import { FormsModule } from '@angular/forms'; // Provides forms-related functionality such as two-way data binding
import { HttpClientModule } from '@angular/common/http'; // Imports HttpClientModule for making HTTP requests (module version of HttpClient)

@Component({
  selector: 'app-shap-8',
  imports: [
    FormsModule,
    HttpClientModule
],
  templateUrl: './shap-8.component.html',
  styleUrl: './shap-8.component.css'
})
export class Shap8Component {
  modelNames: string[] = [];
  featureName: string = '';
  featureValue: number = 0;
  graphicsUrls: string[] = [];
  errorMessage: string = '';  // Variável de erro
  graphicsLoading: boolean = false;

  private apiUrl = 'http://127.0.0.5:5000/api_shap_8';

  constructor(private http: HttpClient) {}

  ngOnInit(): void {
    this.fetchFeatures();
  }

  fetchFeatures(): void {
    const url = `${this.apiUrl}/features`;
    this.http.get<{ Features: string[] }>(url).subscribe({
      next: (response) => {
        this.modelNames = response.Features;
        if (this.modelNames.length > 0) {
          this.featureName = this.modelNames[0]; // Seleciona a primeira feature
          this.fetchGraphics(); // Carrega os gráficos pela primeira vez
        }
      },
      error: (error) => {
        console.error('Error fetching features:', error);
        this.errorMessage = 'An error occurred while fetching features.';
      }
    });
  }

  fetchGraphics(): void {
    if (this.featureValue === undefined || this.featureValue < 0) {
      this.errorMessage = 'Invalid instance value. Please enter a valid number.';
      this.graphicsUrls = [];
      return;  // Para a execução caso o valor da instância não seja válido
    }

    this.graphicsLoading = true;
    const url = `${this.apiUrl}/return_name_graphics/${this.featureName}/${this.featureValue}`;
    this.http.get<{ graphics: string[] }>(url).subscribe({
      next: (response) => {
        if (response.graphics && response.graphics.length > 0) {
          this.graphicsUrls = response.graphics.map(
            (graphic) => `${this.apiUrl}/files/${graphic}`
          );
          this.errorMessage = ''; // Limpa a mensagem de erro
        } else {
          this.graphicsUrls = [];
          this.errorMessage = 'No graphics available for the selected instance value.';
        }
      },
      error: (error) => {
        console.error('Error fetching graphics:', error);
        this.errorMessage = `Error fetching graphics for instance value ${this.featureValue}. Please try again with another value.`;
        this.graphicsUrls = [];
      },
      complete: () => {
        this.graphicsLoading = false;
      }
    });
  }
}