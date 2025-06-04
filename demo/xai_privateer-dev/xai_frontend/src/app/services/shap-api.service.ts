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

  readonly endpointSHAPTimeseriesAPI = "http://127.0.0.1:5002/api_shap"

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
    this.shapReport = MockData.mockShapReport
  }

  ngOnInit() {
  }

  loadSHAPData(datasetName: string, modelName: string) {
    return this.http.get(`${this.endpointSHAPTimeseriesAPI}/send_data_request/${datasetName}/${modelName}`)
  }

  calculateSHAPelyValues(instance: number) {
    // this.shapReportState.set(RequestState.Initiated)
    return this.http.get(`${this.endpointSHAPTimeseriesAPI}/perform_shap_calculations/${instance}`)
    // .subscribe({
    //   next: (data: any) => {
    //     this.shapReport = data.report
    //     localStorage.setItem("shap_report", JSON.stringify(data.report))
    //     this.shapReportState.set(RequestState.Completed)
    //     this.featureValue_shap = instance
    //   },
    //   error: (err) => {
    //     this.shapReportState.set(RequestState.Error)
    //   }
    // })
  }

  generateSHAPGraphics() {
    // this.shapGraphicsState.set(RequestState.Initiated)
    return this.http.get(`${this.endpointSHAPTimeseriesAPI}/generation_graphics`)
    // .subscribe({
    //   next: (data) => {
    //     this.shapGraphicsState.set(RequestState.Completed)
    //   },
    //   error: (err) => {
    //     this.shapGraphicsState.set(RequestState.Error)
    //   }
    // })
  }

  generateSHAPExplanation(datasetName: string, modelName: string, instance: number) {
    return this.http.get(`${this.endpointSHAPTimeseriesAPI}/send_data_request/${datasetName}/${modelName}`)
    .pipe(
      map((message) => console.log(message)),
      mergeMap( () => this.http.get(`${this.endpointSHAPTimeseriesAPI}/perform_shap_calculations/${instance}`)),
      map((message) => console.log(message)),
      mergeMap( () => this.http.get(`${this.endpointSHAPTimeseriesAPI}/generation_graphics`))
    )
  }

  getGraphicsFromServer() {
    return this.http.get<{ files: string[] }>(`${this.endpointSHAPTimeseriesAPI}/files`)
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


}


