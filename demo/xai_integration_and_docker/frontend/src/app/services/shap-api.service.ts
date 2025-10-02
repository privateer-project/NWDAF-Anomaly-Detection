import { HttpClient } from '@angular/common/http';
import { Injectable, signal } from '@angular/core';
import { RequestState } from '../models/utils';
import { LimeApiService } from './lime-api.service';
import { MockData } from '../models/mockData';
import { map, mergeMap } from 'rxjs';
import { error } from 'console';

@Injectable({
  providedIn: 'root'
})
export class ShapApiService {

  // Consolidated SHAP endpoint under backend port 5000 and /api
  readonly endpointSHAPTimeseriesAPI = "/api/shap"

  private shapDataState = signal<RequestState>(RequestState.Not_Initiated)
  readonly shapDataStateSignal = this.shapDataState.asReadonly()

  private shapReportState = signal<RequestState>(RequestState.Not_Initiated)
  readonly shapReportStateSignal = this.shapReportState.asReadonly()

  private shapGraphicsState = signal<RequestState>(RequestState.Not_Initiated)
  readonly shapGraphicsStateSignal = this.shapGraphicsState.asReadonly()

  shapReport: any
  featureValue_shap: number = 0;

  constructor(private http: HttpClient) {
    // this.shapReport =JSON.parse(localStorage.getItem("shap_report")||"")
    try {
      console.log('ShapApiService constructor - MockData:', MockData);
      console.log('ShapApiService constructor - MockData.mockShapReport:', MockData.mockShapReport);
      this.shapReport = MockData.mockShapReport
      console.log('ShapApiService constructor - shapReport initialized:', this.shapReport);
    } catch (error) {
      console.error('Error initializing shapReport:', error);
      this.shapReport = {};
    }
  }

  ngOnInit() {
  }

  // Unified calculate endpoint expects the input tensor payload
  calculateSHAP(payload: any) {
    return this.http.post(`${this.endpointSHAPTimeseriesAPI}/calculate`, payload)
  }

  readonly labels = [
    "dl_bitrate",
    "dl_retx",
    "dl_tx",
    "ul_bitrate",
    "ul_mcs",
    "ul_retx",
    "ul_tx",
    "turbo_decoder_avg"
  ]


  // --- New: fetch last SHAP result from backend and helpers to map into 12x8 ---
  getLastResult() {
    // Endpoint base already includes /api/shap
    return this.http.get(`${this.endpointSHAPTimeseriesAPI}/last_result`)
  }

  // Map backend shap_values (8 features) into a 12x8 matrix by repeating across 12 windows
  mapShapLastResultToMatrix(response: any, windows: number = 12): number[][] {
    console.log('mapShapLastResultToMatrix input:', response);
    // Prefer backend-declared ordering
    const shapValuesObj = response?.shap_values || {}
    console.log('shapValuesObj:', shapValuesObj);
    const features: string[] = Array.isArray(response?.feature_names) && response.feature_names.length
      ? response.feature_names
      : Object.keys(shapValuesObj)
    console.log('features:', features);
    const row: number[] = features.map((f) => Number(shapValuesObj[f] ?? 0))
    console.log('row:', row);
    console.log('row values check:', row.map((val, i) => `${features[i]}: ${val}`));
    const matrix: number[][] = Array.from({ length: windows }, () => [...row])
    console.log('matrix:', matrix);
    console.log('matrix sample (first row):', matrix[0]);
    return matrix
  }


}


