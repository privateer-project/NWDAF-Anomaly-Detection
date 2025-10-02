import { Component, Inject, PLATFORM_ID } from '@angular/core'; // Imports the base Angular component class for building components
 // Provides common Angular directives such as ngIf and ngFor
import { FormsModule } from '@angular/forms'; // Provides forms-related functionality such as two-way data binding
import { ShapApiService } from '../services/shap-api.service';
import { isPlatformBrowser } from '@angular/common';

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

  constructor(private shapService:ShapApiService, @Inject(PLATFORM_ID) private platformId: Object) {}

  ngOnInit(): void {
    // Only make HTTP requests in the browser, not during SSR
    if (isPlatformBrowser(this.platformId)) {
      this.fetchFeatures(); // Obter a lista de features
    }
  }

  // Método para carregar a lista de features
 fetchFeatures(): void {
  this.modelNames = ['shap_summary','shap_water_fall','shap_decision']; // Futuramente virá do backend

  if (this.modelNames.length > 0) {
    this.featureName = this.modelNames[0];
    this.fetchGraphics();
  }
}


  // Método placeholder: geração/serving de gráficos não está disponível no backend atual
fetchGraphics(): void {
  this.graphicsUrls = []
  this.noGraphicsMessage = 'Graphics endpoint is not available in the current backend version.'
}
}
