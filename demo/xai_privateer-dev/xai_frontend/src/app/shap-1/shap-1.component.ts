import { Component } from '@angular/core'; // Imports the base Angular component class for building components
 // Provides common Angular directives such as ngIf and ngFor
import { FormsModule } from '@angular/forms'; // Provides forms-related functionality such as two-way data binding
import { ShapApiService } from '../services/shap-api.service';

@Component({
  selector: 'app-shap-1',
  imports: [
    FormsModule
],
  templateUrl: './shap-1.component.html',
  styleUrl: './shap-1.component.css'
})
export class Shap1Component {
  featureName: string = ''; // Nome da feature padrão
  featureValue: number = 0; // Valor padrão para a instância
  graphicsUrl: string = ''; // URL do gráfico gerado
  modelNames: string[] = []; // Lista de features carregadas
  graphicsUrls: string[] = []; // Lista de URLs dos gráficos
  noGraphicsMessage: string = ''; // Mensagem de erro para gráficos inexistentes

  constructor(private shapService:ShapApiService) {}

  ngOnInit(): void {
    this.fetchFeatures(); // Obter a lista de features
  }

  // Método para carregar a lista de features
 fetchFeatures(): void {
  this.modelNames = ['shap_summary','shap_water_fall','shap_decision']; // Futuramente virá do backend

  if (this.modelNames.length > 0) {
    this.featureName = this.modelNames[0];
    this.fetchGraphics();
  }
}


  // Método para carregar gráficos
fetchGraphics(): void {
  // const url = `http://127.0.0.4:5000/api_shap/files`;

  this.shapService.getGraphicsFromServer().subscribe({
    next: (response) => {
      const files = response.files || [];

      // Filtrar por tipo de gráfico e instância
      this.graphicsUrls = files
        // .filter(file =>
        //   file.startsWith(this.featureName) &&
        //   file.includes(`_${this.featureValue}.png`)
        // )
        .map(file => `${this.shapService.endpointSHAPTimeseriesAPI}/files/${file}`);

      if (this.graphicsUrls.length === 0) {
        this.noGraphicsMessage = 'No graphics available for the selected type and instance.';
      } else {
        this.noGraphicsMessage = '';
      }
    },
    error: (error) => {
      console.error('Error fetching graphics:', error);
      this.noGraphicsMessage = `Error fetching graphics.`;
      this.graphicsUrls = [];
    }
  });
}
}
