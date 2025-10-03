import { Component } from '@angular/core';
import { ShapApiService } from '../../../services/shap-api.service';
import { ChartConfiguration, ChartData, ChartEvent } from 'chart.js';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';

@Component({
  selector: 'app-window',
  imports: [BaseChartDirective, CommonModule],
  templateUrl: './window.component.html',
  styleUrl: './window.component.css'
})
export class WindowComponent {

  shapvalues: number[][]

  featureLabels

  // HeatMap Component
  windowChartDataShap: any
  windowChartDataLime: any
  // data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

  // Chart Data
  public barChartType = 'bar' as const;


  public barChartOptions: ChartConfiguration<'bar'>['options'] = {}
  public barChartDataShap: ChartData<'bar'> = { labels: [], datasets: [] }

  constructor(private shap: ShapApiService) {
    // this.shapvalues = this.convertTo2DArray(this.shap.shapReport.shap_values) as number[][]
    // this.limeValues = this.limeService.fillMissingValuesLimeReport()
    this.shapvalues = []
    this.featureLabels = this.shap.labels

    this.columnLabels = this.shap.labels
    // Delay chart build until backend returns
  }

  ngOnInit(): void {

    // Fetch backend last results and update local charts
    this.shap.getLastResult().subscribe({
      next: (resp: any) => {
        // console.log('SHAP response from backend (window):', resp);
        // const shapMatrix = this.shap.mapShapLastResultToMatrix(resp)
        // console.log('SHAP matrix after mapping (window):', shapMatrix);
        // this.shapvalues = this.transpose(shapMatrix)
        // console.log('SHAP values after transpose (window):', this.shapvalues);
        // rebuild charts
        this.shapvalues = this.convertTo2DArray(resp.shap_values) as number[][]

        const initShapCharts = this.init_feature_data_graphic(this.shapvalues)
        this.barChartDataShap = initShapCharts.barChartData
        this.barChartOptions = initShapCharts.barChartOptions
        this.windowChartDataShap = this.generateFeatureChartData(this.shapvalues, this.columnLabels)
      },
      error: (err) => {
        console.error('Error fetching SHAP data (window):', err);
      }
    })
  }

  private init_feature_data_graphic(data:number[][]){
      let graph_data = this.calculateStats(data)
      let barChartData = {
        labels: Array.from({ length: 12 }, (_, index) => index + 1),
        datasets: [
          { data: graph_data!.means, label: 'Mean' },
          { data: graph_data!.stdDevs, label: 'Standard Deviation' },
        ],
      }
      let barChartOptions = {
        plugins: {
          legend: {
            display: true,
          },
        },
      };
      return {barChartData, barChartOptions}
    }
    
      transpose<T>(matrix: number[][]): number[][] {
        return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
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
    
    
      //---------------------
    
    
      // events
      public chartClicked({
        event,
        active,
      }: {
        event?: ChartEvent;
        active?: object[];
      }): void {
        console.log(event, active);
      }
    
      public chartHovered({
        event,
        active,
      }: {
        event?: ChartEvent;
        active?: object[];
      }): void {
        console.log(event, active);
      }
    
    // Calculate mean and standard deviation for each row
    private calculateStats(data: number[][]): { means: number[]; stdDevs: number[] } {
      if (!data || data.length === 0 || data[0].length === 0) {
        return { means: [], stdDevs: [] };
      }
  
      const numRows = data.length;
  
      // Calculate mean for each row
      const means = data.map(row => {
        const sum = row.reduce((acc, val) => acc + val, 0);
        return sum / row.length;
      });
  
      // Calculate standard deviation for each row
      const stdDevs = data.map((row, index) => {
        const mean = means[index];
        const variance = row.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / row.length;
        return Math.sqrt(variance);
      });
  
      return { means, stdDevs };
    }
  
    private generateFeatureChartData(data: number[][], features:string[]){
      let resp = []
      let data_transposed = this.transpose(data)
      let window = data_transposed[0].length
  
      for(let i=0;i<window ; i++){
        let temp = {
          labels: this.featureLabels,
          datasets: [
            { data: data_transposed[i], label: features[i] ?? `feature_${i}` }  
          ],
      }
      resp.push({data:temp, features:features[i]})
      }
      return resp
    }
    
}
