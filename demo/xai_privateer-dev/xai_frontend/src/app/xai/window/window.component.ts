import { Component } from '@angular/core';
import { BaseChartDirective } from 'ng2-charts';
import { HeatmapComponent } from '../../general-components/heatmap/heatmap.component';
import { ShapApiService } from '../../services/shap-api.service';
import { ChartConfiguration, ChartData, ChartEvent } from 'chart.js';
import { LimeApiService } from '../../services/lime-api.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-window',
  imports: [BaseChartDirective, CommonModule],
  templateUrl: './window.component.html',
  styleUrl: './window.component.css'
})
export class WindowComponent {
  
  shapvalues: number[][]
  limeValues:number [][]

  featureLabels

  // HeatMap Component
  windowChartDataShap
  windowChartDataLime
  // data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

  // Chart Data
  public barChartType = 'bar' as const;

  
  public barChartOptions: ChartConfiguration<'bar'>['options']
  public barChartDataShap: ChartData<'bar'>
  public barChartDataLime: ChartData<'bar'> 
  
    constructor(private shap: ShapApiService, private limeService:LimeApiService) {
        this.shapvalues = this.convertTo2DArray(this.shap.shapReport.shap_values) as number[][]
        this.limeValues = this.limeService.fillMissingValuesLimeReport()
        this.featureLabels = this.shap.labels
    
        this.columnLabels = this.shap.labels
        let graph_data = this.calculateStats(this.shapvalues)
        let initShapCharts = this.init_feature_data_graphic(this.shapvalues)
        this.barChartDataShap = initShapCharts.barChartData
        this.barChartOptions = initShapCharts.barChartOptions
        let initLimeCharts = this.init_feature_data_graphic(this.limeValues)
        this.barChartDataLime = initLimeCharts.barChartData
        this.windowChartDataShap=this.generateFeatureChartData(this.shapvalues,this.columnLabels)
        this.windowChartDataLime=this.generateFeatureChartData(this.limeValues,this.columnLabels)
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
          { data: data_transposed[i], label: features[i] }  
        ],
    }
    resp.push({data:temp, features:features[i]})
    }
    return resp
  }

}
