import { Component } from '@angular/core';
import { ShapApiService } from '../../../services/shap-api.service';
import { CommonModule } from '@angular/common';
import { HeatmapComponent } from '../../../general-components/heatmap/heatmap.component';

@Component({
  selector: 'app-timeseries',
  imports: [CommonModule, HeatmapComponent],
  templateUrl: './timeseries.component.html',
  styleUrl: './timeseries.component.css'
})
export class TimeseriesComponent {

  shap_data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

limeGraphicsUrls: string[] = [];



  featureName: string = ''; // Nome da feature padrão
  //featureValue: number = 0; // Valor padrão para a instância
  graphicsUrl: string = ''; // URL do gráfico gerado
  modelNames: string[] = []; // Lista de features carregadas
  graphicsUrls: string[] = []; // Lista de URLs dos gráficos
  noGraphicsMessage: string = ''; // Mensagem de erro para gráficos inexistentes

  constructor(private shapService: ShapApiService) {
    console.log('Timeseries Constructor - shapService.shapReport:', this.shapService.shapReport);
    this.columnLabels = this.shapService.labels
    console.log('Timeseries Constructor - columnLabels:', this.columnLabels);

    // Initialize with empty arrays - will be populated from backend
    this.shap_data = []
  }


  ngOnInit(): void {
      // Fetch backend last results and update local data
      this.shapService.getLastResult().subscribe({
        next: (resp: any) => {
          console.log('SHAP response from backend (timeseries):', resp);
          this.shap_data = this.shapService.mapShapLastResultToMatrix(resp)
          console.log('SHAP data after mapping (timeseries):', this.shap_data);
          console.log('SHAP data dimensions (timeseries):', this.shap_data?.length, 'x', this.shap_data?.[0]?.length);
          console.log('SHAP data sample (timeseries):', this.shap_data?.[0]?.slice(0, 3));
          console.log('SHAP data for heatmap (timeseries):', this.shap_data);
        },
        error: (err) => {
          console.error('Error fetching SHAP data (timeseries):', err);
        }
      })
    }

}
