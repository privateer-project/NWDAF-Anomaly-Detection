import { Component } from '@angular/core';
import { HeatmapComponent } from '../../general-components/heatmap/heatmap.component';
import { ShapApiService } from '../../services/shap-api.service';
import { LimeApiService } from '../../services/lime-api.service';
import { CommonModule } from '@angular/common';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser'; // Para funcionar os graficos com extensão html

@Component({
  selector: 'app-timeseries',
  imports: [CommonModule, HeatmapComponent],
  templateUrl: './timeseries.component.html',
  styleUrl: './timeseries.component.css'
})
export class TimeseriesComponent {

  data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

  limeFeaturesData:number[][]

limeGraphicsUrls: string[] = [];

limeHtmlUrlSafe: SafeResourceUrl | null = null; // Para graficos html - lime
shapHtmlUrlSafe: SafeResourceUrl | null = null;


  featureName: string = ''; // Nome da feature padrão
  //featureValue: number = 0; // Valor padrão para a instância
  graphicsUrl: string = ''; // URL do gráfico gerado
  modelNames: string[] = []; // Lista de features carregadas
  graphicsUrls: string[] = []; // Lista de URLs dos gráficos
  noGraphicsMessage: string = ''; // Mensagem de erro para gráficos inexistentes

  constructor(private shapService: ShapApiService, public lime:LimeApiService, private sanitizer: DomSanitizer) {
    this.data = this.convertTo2DArray(this.shapService.shapReport.shap_values) as number[][]
    this.columnLabels = this.shapService.labels
    this.limeFeaturesData = this.lime.fillMissingValuesLimeReport()
  }

  ngOnInit(): void {
    this.fetchFeatures(); // Obter a lista de features
    this.fetchLimeGraphics();
  }

  // Método para carregar a lista de features
 fetchFeatures(): void {
  this.modelNames = [
  'shap_summary_bar',
  'shap_summary_dot',
  'shap_decision',
  'shap_dependence',
  'shap_waterfall',
  'shap_force'
];
 // Futuramente virá do backend

  if (this.modelNames.length > 0) {
    this.featureName = this.modelNames[0];
    this.fetchGraphics();
  }
}


  convertTo2DArray(obj: any, groupSize = 8) {
    // Get all values from the object in order
    const values = Object.values(obj);

    // Create 2D array by grouping consecutive values
    const result = [];

    for (let i = 0; i < values.length; i += groupSize) {
      // Extract a slice of 'groupSize' elements
      const group = values.slice(i, i + groupSize);
      result.push(group);
    }

    return result;
  }

   // Método para carregar gráficos
fetchGraphics_2_so_grafico_png(): void {
  // const url = `http://127.0.0.4:5000/api_shap/files`;

  this.shapService.getGraphicsFromServer().subscribe({
    next: (response) => {
      const files = response.files || [];

      // Filtrar por tipo de gráfico e instância
      //this.graphicsUrls = files
      //  .filter(file =>
        //   file.startsWith(this.featureName) &&
       //   file.includes(`_${this.featureValue}.png`)
       // )
       // .map(file => `${this.shapService.endpointSHAPTimeseriesAPI}/files/${file}`);

      this.graphicsUrls = files
      .filter(file =>
        file.includes(`_instance_${this.shapService.featureValue_shap}`) && file.endsWith('.png')
      )
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

fetchGraphics(): void {
  this.shapService.getGraphicsFromServer().subscribe({
    next: (response) => {
      const files = response.files || [];

      // PNGs
      this.graphicsUrls = files
        .filter(file =>
          file.includes(`_instance_${this.shapService.featureValue_shap}`) && file.endsWith('.png')
        )
        .map(file => `${this.shapService.endpointSHAPTimeseriesAPI}/files/${file}`);

      // HTML
      const htmlFile = files.find(file =>
        file.includes(`_instance_${this.shapService.featureValue_shap}`) && file.endsWith('.html')
      );
      if (htmlFile) {
        const fullUrl = `${this.shapService.endpointSHAPTimeseriesAPI}/files/${htmlFile}`;
        this.shapHtmlUrlSafe = this.sanitizer.bypassSecurityTrustResourceUrl(fullUrl);
      } else {
        this.shapHtmlUrlSafe = null;
      }

      this.noGraphicsMessage = this.graphicsUrls.length === 0 && !this.shapHtmlUrlSafe
        ? 'No graphics available for the selected type and instance.'
        : '';
    },
    error: (error) => {
      console.error('Error fetching SHAP graphics:', error);
      this.graphicsUrls = [];
      this.shapHtmlUrlSafe = null;
      this.noGraphicsMessage = `Error fetching graphics.`;
    }
  });
}


fetchLimeGraphics_2_so_grafico_png(): void {
  this.lime.getLimeGraphicsFiles().subscribe({
    next: (response) => {
      const files = response.files || [];

      this.limeGraphicsUrls = files
        .filter((file: string) =>
          file.includes(`_instance_${this.lime.featureValue_lime}`) && file.endsWith('.png')
        )
        .map((file: string) => `${this.lime.endpointLIMETimeseriesAPI}/files/${file}`);
    },
    error: (error) => {
      console.error('Erro ao obter gráficos LIME:', error);
      this.limeGraphicsUrls = [];
    }
  });
}

fetchLimeGraphics(): void {
  this.lime.getLimeGraphicsFiles().subscribe({
    next: (response) => {
      const files = response.files || [];

      // PNGs
      this.limeGraphicsUrls = files
        .filter((file: string) =>
          file.includes(`_instance_${this.lime.featureValue_lime}`) && file.endsWith('.png')
        )
        .map((file: string) => `${this.lime.endpointLIMETimeseriesAPI}/files/${file}`);

      // HTML (assume que há apenas um por instância)
      const htmlFile = files.find((file: string) =>
        file.includes(`_instance_${this.lime.featureValue_lime}`) && file.endsWith('.html')
      );
      if (htmlFile) {
        const fullUrl = `${this.lime.endpointLIMETimeseriesAPI}/files/${htmlFile}`;
        this.limeHtmlUrlSafe = this.sanitizer.bypassSecurityTrustResourceUrl(fullUrl);
      } else {
        this.limeHtmlUrlSafe = null;
      }
    },
    error: (error) => {
      console.error('Erro ao obter gráficos LIME:', error);
      this.limeGraphicsUrls = [];
      this.limeHtmlUrlSafe = null;
    }
  });
}


}