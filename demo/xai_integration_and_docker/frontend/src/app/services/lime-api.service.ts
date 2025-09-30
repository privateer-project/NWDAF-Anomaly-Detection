import { HttpClient } from '@angular/common/http';
import { Injectable, signal } from '@angular/core';
import { RequestState } from '../models/utils';
import { MockData } from '../models/mockData';

@Injectable({
  providedIn: 'root'
})
export class LimeApiService {

  readonly endpointLIMETimeseriesAPI = "http://127.0.0.1:5000/api/lime"

  limeReport:any
  featureValue_lime: number = 0;

  private limeDataState=signal<RequestState>(RequestState.Not_Initiated)
  readonly limeDataStateSignal = this.limeDataState.asReadonly()

  private limeReportState=signal<RequestState>(RequestState.Not_Initiated)
  readonly limeReportStateSignal = this.limeReportState.asReadonly()

  private limeGraphicsState=signal<RequestState>(RequestState.Not_Initiated)
  readonly limeGraphicsStateSignal = this.limeGraphicsState.asReadonly()

  constructor(private http:HttpClient) {
    this.limeReport = MockData.mockLimeReport
  }

  // Unified backend: calculate via POST payload
  calculateLIME(payload: any){
    return this.http.post(`${this.endpointLIMETimeseriesAPI}/calculate`, payload)
  }


  public fillMissingValuesLimeReport(){
    let data = this.limeReport.contribution
    let limeDenseReport = Array.from({ length: 12 }, () => Array(8).fill(0));
    for(let elem of data){
      // for (const [key, value] of Object.entries(elem)){
        const parsed = this.parseKey(elem.feature);
        if(!parsed){
          continue
        }
        const { featureName, rowIndex } = parsed;
        console.log("ei "+featureName+" "+rowIndex)
        const colIndex = this.labels.indexOf(featureName)
        limeDenseReport[rowIndex][colIndex] = elem.value;
      // }
    }
    return limeDenseReport
  }

  public parseKey(key: string): ParsedKey | null {
  // Extract feature name and row index from key like "feature_a_1" or "feature_b_2"
  const match = key.match(/^(.+)_(\d+)$/);
  if (!match){ console.log("umatched") ; return null;}
  return {
    featureName: match[1],
    rowIndex: parseInt(match[2], 10)
  };
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

  // --- New: fetch last LIME result and map to 12x8 matrix ---
  getLastResult(){
    // Endpoint base already includes /api/lime
    return this.http.get(`${this.endpointLIMETimeseriesAPI}/last_result`)
  }

  // Map backend lime_values (8 features) into a 12x8 matrix by repeating across 12 windows
  mapLimeLastResultToMatrix(response: any, windows: number = 12): number[][] {
    // Prefer backend-declared ordering
    const limeValuesObj = response?.lime_values || {}
    const features: string[] = Array.isArray(response?.feature_names) && response.feature_names.length
      ? response.feature_names
      : Object.keys(limeValuesObj)
    const row: number[] = features.map((f) => Number(limeValuesObj[f] ?? 0))
    const matrix: number[][] = Array.from({ length: windows }, () => [...row])
    return matrix
  }
}

interface ParsedKey {
  featureName: string;
  rowIndex: number;
}
