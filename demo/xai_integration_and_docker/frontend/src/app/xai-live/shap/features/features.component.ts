import { Component } from '@angular/core';
import { ShapApiService } from '../../../services/shap-api.service';
import { ChartConfiguration, ChartData, ChartEvent } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-features',
  imports: [CommonModule, BaseChartDirective],
  templateUrl: './features.component.html',
  styleUrl: './features.component.css'
})
export class FeaturesComponent {
  shapvalues: any

  // HeatMap Component
  featureChartDataShap: any
  // data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

  // Chart Data
  public barChartType = 'bar' as const;

  public barChartOptions: ChartConfiguration<'bar'>['options'] = {}
  public barChartDataShap: ChartData<'bar'> = { labels: [], datasets: [] }

  // Only render charts after backend returns data (split by SHAP/LIME)
  // private backendLoaded: boolean = false
  private shapLoaded: boolean = false

  constructor(private shap: ShapApiService) {
    console.log('Constructor - shap.shapReport:', this.shap.shapReport);
    console.log('Constructor - shap.shapReport.shap_values:', this.shap.shapReport?.shap_values);

    if (this.shap.shapReport?.shap_values) {
      this.shapvalues = this.convertTo2DArray(this.shap.shapReport.shap_values) as number[][]
    } else {
      console.log('Constructor - shap_values not available, using empty array');
      this.shapvalues = [];
    }
    this.columnLabels = this.shap.labels
    //Initialize with existing mock/local values to preserve current behavior
    this.refreshCharts(this.shapvalues)
  }

  ngOnInit(): void {
    // Fetch last results from backend and update charts when available
    this.shap.getLastResult().subscribe({
      next: (resp: any) => {
        this.shapvalues = this.convertTo2DArray(resp.shap_values)
        // console.log('SHAP response from backend (features):', resp);
        // const shapMatrix = this.shap.mapShapLastResultToMatrix(resp)
        // console.log('SHAP matrix after mapping (features):', shapMatrix);
        // console.log('SHAP matrix dimensions (features):', shapMatrix?.length, 'x', shapMatrix?.[0]?.length);
        // this.shapvalues = shapMatrix
        this.shapLoaded = true
        console.log('About to call refreshShapCharts with:', this.shapvalues);
        this.refreshCharts(this.shapvalues)
      },
      error: (err) => {
        console.error('Error fetching SHAP data (features):', err);
      }
    })
  }

  private refreshCharts(shapData: number[][]){
    // if (!this.backendLoaded) { return }
    const initShapCharts = this.init_feature_data_graphic(shapData)
    this.barChartDataShap = initShapCharts.barChartData
    this.barChartOptions = initShapCharts.barChartOptions
    this.featureChartDataShap=this.generateFeatureChartData(shapData,this.columnLabels)
  }

  private init_feature_data_graphic(data:number[][]){
    console.log('init_feature_data_graphic called with data:', data);
    let graph_data = this.calculateStats(data)
    console.log('init_feature_data_graphic: graph_data:', graph_data);
    
    if (!graph_data) {
      console.log('init_feature_data_graphic: graph_data is null, returning empty chart');
      return {
        barChartData: { labels: [], datasets: [] },
        barChartOptions: { plugins: { legend: { display: true } } }
      };
    }
    
    let barChartData = {
      labels: this.columnLabels,
      datasets: [
        { data: graph_data.means, label: 'Mean' },
        { data: graph_data.stdDevs, label: 'Standard Deviation' },
      ],
    }
    let barChartOptions = {
      plugins: {
        legend: {
          display: true,
        },
      },
    };
    console.log('init_feature_data_graphic: barChartData:', barChartData);
    return {barChartData, barChartOptions}
  }
  
  private calculateStats(data: number[][]) {
    console.log('calculateStats called with data:', data);
    if (!data || data.length === 0 || data[0].length === 0) {
      console.log('calculateStats: invalid data, returning undefined');
      return undefined;
    }

    const numRows = data.length;
    const numCols = data[0].length;
    let means = []
    let stdDevs = []

    // Calculate mean for each column
    means = Array(numCols).fill(0).map((_, colIndex) => {
      const column = data.map(row => row[colIndex]);
      const sum = column.reduce((acc, val) => acc + val, 0);
      return sum / numRows;
    });

    // Calculate standard deviation for each column
    stdDevs = Array(numCols).fill(0).map((_, colIndex) => {
      const column = data.map(row => row[colIndex]);
      const mean = means[colIndex];
      const variance = column.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / numRows;
      return Math.sqrt(variance);
    });

    return { means, stdDevs }
  }

  private generateFeatureChartData(data: number[][], features:string[]){
    let resp = []
    let data_transposed = this.transpose(data)

    for(let i=0;i<features.length; i++){
      let temp = {
        labels: [1,2,3,4,5,6,7,8,9,10,11,12],
        datasets: [
          { data: data_transposed[i], label: features[i] }  
        ],
    }
    resp.push({data:temp, features:features[i]})
    }

    return resp
  }

  transpose<T>(matrix: T[][]): T[][] {
    return matrix[0].map((_, colIndex) => matrix.map(row => row[colIndex]));
  }

  

  convertTo2DArray(obj: any, groupSize = 8) {
    console.log('convertTo2DArray called with obj:', obj);
    if (!obj) {
      console.log('convertTo2DArray: obj is null/undefined, returning empty array');
      return [];
    }
    
    // Get all values from the object in order
    const values = Object.values(obj);
    console.log('convertTo2DArray: values extracted:', values);

    // Create 2D array by grouping consecutive values
    const result = [];

    for (let i = 0; i < values.length; i += groupSize) {
      // Extract a slice of 'groupSize' elements
      const group = values.slice(i, i + groupSize);
      result.push(group);
    }

    console.log('convertTo2DArray: result:', result);
    return result;
  }

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
    
}
