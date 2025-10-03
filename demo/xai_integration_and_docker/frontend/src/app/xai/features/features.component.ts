import { Component, ViewChild, Inject, PLATFORM_ID } from '@angular/core';
import { ShapApiService } from '../../services/shap-api.service';
import { HeatmapComponent } from '../../general-components/heatmap/heatmap.component';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartData, ChartEvent } from 'chart.js';
import { LimeApiService } from '../../services/lime-api.service';
import { CommonModule, isPlatformBrowser } from '@angular/common';
import { SidebarComponent } from '../../sidebar/sidebar.component';

@Component({
  selector: 'app-features',
  imports: [CommonModule, BaseChartDirective, SidebarComponent],
  templateUrl: './features.component.html',
  styleUrl: './features.component.css'
})
export class FeaturesComponent {

  shapvalues: any
  limeValues:number [][]

  // HeatMap Component
  featureChartDataShap: any
  featureChartDataLime: any
  // data: number[][]
  columnLabels: string[]
  colors: string[] = ['#f5f5f5', '#e0f7fa', '#80deea', '#00acc1', '#006064'];
  threshold: number = 50;

  // Chart Data
  public barChartType = 'bar' as const;

  
  public barChartOptions: ChartConfiguration<'bar'>['options'] = {}
  public barChartDataShap: ChartData<'bar'> = { labels: [], datasets: [] }
  public barChartDataLime: ChartData<'bar'> = { labels: [], datasets: [] } 

  // Only render charts after backend returns data (split by SHAP/LIME)
  // private backendLoaded: boolean = false
  private shapLoaded: boolean = false
  private limeLoaded: boolean = false

  constructor(private shap: ShapApiService, private limeService:LimeApiService, @Inject(PLATFORM_ID) private platformId: Object) {
    console.log('Constructor - shap.shapReport:', this.shap.shapReport);
    console.log('Constructor - shap.shapReport.shap_values:', this.shap.shapReport?.shap_values);
    
    if (this.shap.shapReport?.shap_values) {
      this.shapvalues = this.convertTo2DArray(this.shap.shapReport.shap_values) as number[][]
    } else {
      console.log('Constructor - shap_values not available, using empty array');
      this.shapvalues = [];
    }
    
    this.limeValues = this.limeService.fillMissingValuesLimeReport()
    this.columnLabels = this.shap.labels
    console.log('Constructor - columnLabels initialized:', this.columnLabels);
    console.log('Constructor - shapvalues initialized:', this.shapvalues);

    // Initialize with existing mock/local values to preserve current behavior
    // this.refreshCharts(this.shapvalues, this.limeValues)
  }

  ngOnInit(): void {
    // Only make HTTP requests in the browser, not during SSR
    if (isPlatformBrowser(this.platformId)) {
      // Fetch last results from backend and update charts when available
      this.shap.getLastResult().subscribe({
        next: (resp: any) => {
          console.log('SHAP response from backend (features):', resp);
          //const shapMatrix = this.shap.mapShapLastResultToMatrix(resp)
          this.shapvalues = this.convertTo2DArray(resp.shap_values)

          // const shapMatrix = this.shap.mapShapLastResultToMatrix(resp)
          // console.log('SHAP matrix after mapping (features):', shapMatrix);
          // console.log('SHAP matrix dimensions (features):', shapMatrix?.length, 'x', shapMatrix?.[0]?.length);
          // this.shapvalues = shapMatrix
          this.shapLoaded = true
          console.log('About to call refreshShapCharts with:', this.shapvalues);
          this.refreshShapCharts(this.shapvalues)
        },
        error: (err) => {
          console.error('Error fetching SHAP data (features):', err);
        }
      })

      this.limeService.getLastResult().subscribe({
        next: (resp: any) => {
          this.limeService.limeReport = resp
          this.limeValues = this.limeService.fillMissingValuesLimeReport()
          this.limeLoaded = true
          // this.refreshCharts(this.shapvalues, this.limeValues)
          //this.refreshLimeCharts(this.limeValues)
          const initLimeCharts = this.init_feature_data_graphic(this.limeValues)
          this.barChartOptions = initLimeCharts.barChartOptions
          this.barChartDataLime = initLimeCharts.barChartData
          this.featureChartDataLime=this.generateFeatureChartData(this.limeValues,this.columnLabels)
        },
        error: () => {}
      })
    }
  }

  private refreshCharts(shapData: number[][], limeData: number[][]){
    // if (!this.backendLoaded) { return }
    const initShapCharts = this.init_feature_data_graphic(shapData)
    this.barChartDataShap = initShapCharts.barChartData
    this.barChartOptions = initShapCharts.barChartOptions
    const initLimeCharts = this.init_feature_data_graphic(limeData)
    this.barChartDataLime = initLimeCharts.barChartData
    this.featureChartDataShap=this.generateFeatureChartData(shapData,this.columnLabels)
    this.featureChartDataLime=this.generateFeatureChartData(limeData,this.columnLabels)
  }

  private refreshShapCharts(shapData: number[][]){
    console.log('refreshShapCharts called with data:', shapData);
    console.log('shapLoaded:', this.shapLoaded);
    console.log('columnLabels:', this.columnLabels);
    if (!this.shapLoaded) { 
      console.log('SHAP not loaded, returning');
      return 
    }
    
    if (!shapData || shapData.length === 0) {
      console.log('refreshShapCharts: shapData is empty, returning');
      return;
    }
    
    const initShapCharts = this.init_feature_data_graphic(shapData)
    console.log('initShapCharts:', initShapCharts);
    this.barChartDataShap = initShapCharts.barChartData
    this.featureChartDataShap=this.generateFeatureChartData(shapData,this.columnLabels)
    console.log('barChartDataShap after refresh:', this.barChartDataShap);
    console.log('featureChartDataShap after refresh:', this.featureChartDataShap);
  }

  private refreshLimeCharts(limeData: number[][]){
    if (!this.limeLoaded) { return }
    const initLimeCharts = this.init_feature_data_graphic(limeData)
    this.barChartDataLime = initLimeCharts.barChartData
    this.featureChartDataLime=this.generateFeatureChartData(limeData,this.columnLabels)
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

  private init_feature_data_lime_graphic(data:number[][]){
    let graph_data = this.calculateStats(data)
    let barChartData = {
      labels: this.columnLabels,
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

}
