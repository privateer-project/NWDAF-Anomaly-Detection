import { Injectable, signal } from '@angular/core';
import { XAIApiService, XAIInput, LIMEResponse, SHAPResponse } from './xai-api.service';
import { Observable } from 'rxjs';

export enum RequestState {
  Not_Initiated = 'not_initiated',
  Loading = 'loading',
  Success = 'success',
  Error = 'error'
}

@Injectable({
  providedIn: 'root'
})
export class DataService {
  // State signals
  private modelState = signal<RequestState>(RequestState.Not_Initiated);
  private datasetState = signal<RequestState>(RequestState.Not_Initiated);
  private limeState = signal<RequestState>(RequestState.Not_Initiated);
  private shapState = signal<RequestState>(RequestState.Not_Initiated);

  // Data signals
  private modelInfo = signal<any>(null);
  private datasetInfo = signal<any>(null);
  private limeData = signal<LIMEResponse | null>(null);
  private shapData = signal<SHAPResponse | null>(null);

  // Readonly signals for components
  readonly modelStateSignal = this.modelState.asReadonly();
  readonly datasetStateSignal = this.datasetState.asReadonly();
  readonly limeStateSignal = this.limeState.asReadonly();
  readonly shapStateSignal = this.shapState.asReadonly();

  readonly modelInfoSignal = this.modelInfo.asReadonly();
  readonly datasetInfoSignal = this.datasetInfo.asReadonly();
  readonly limeDataSignal = this.limeData.asReadonly();
  readonly shapDataSignal = this.shapData.asReadonly();

  constructor(private xaiApi: XAIApiService) {
    this.initializeData();
  }

  private initializeData(): void {
    this.loadModelInfo();
    this.loadDatasetInfo();
  }

  // Model Management
  loadModelInfo(): void {
    this.modelState.set(RequestState.Loading);
    this.xaiApi.getModelInfo().subscribe({
      next: (info) => {
        this.modelInfo.set(info);
        this.modelState.set(RequestState.Success);
      },
      error: (error) => {
        console.error('Error loading model info:', error);
        this.modelState.set(RequestState.Error);
      }
    });
  }

  uploadModel(file: File): Observable<any> {
    this.modelState.set(RequestState.Loading);
    return this.xaiApi.uploadModel(file);
  }

  // Dataset Management
  loadDatasetInfo(): void {
    this.datasetState.set(RequestState.Loading);
    this.xaiApi.getDatasetInfo().subscribe({
      next: (info) => {
        this.datasetInfo.set(info);
        this.datasetState.set(RequestState.Success);
      },
      error: (error) => {
        console.error('Error loading dataset info:', error);
        this.datasetState.set(RequestState.Error);
      }
    });
  }

  uploadDataset(file: File): Observable<any> {
    this.datasetState.set(RequestState.Loading);
    return this.xaiApi.uploadDataset(file);
  }

  // LIME Analysis
  calculateLIME(data: XAIInput): void {
    this.limeState.set(RequestState.Loading);
    this.xaiApi.calculateLIME(data).subscribe({
      next: (result) => {
        this.limeData.set(result);
        this.limeState.set(RequestState.Success);
      },
      error: (error) => {
        console.error('Error calculating LIME:', error);
        this.limeState.set(RequestState.Error);
      }
    });
  }

  // SHAP Analysis
  calculateSHAP(data: XAIInput): void {
    this.shapState.set(RequestState.Loading);
    this.xaiApi.calculateSHAP(data).subscribe({
      next: (result) => {
        this.shapData.set(result);
        this.shapState.set(RequestState.Success);
      },
      error: (error) => {
        console.error('Error calculating SHAP:', error);
        this.shapState.set(RequestState.Error);
      }
    });
  }

  // Utility methods
  generateRandomData(shape: number[] = [1, 12, 8]): XAIInput {
    const data: number[][] = [];
    for (let i = 0; i < shape[1]; i++) {
      const row: number[] = [];
      for (let j = 0; j < shape[2]; j++) {
        row.push(Math.random() * 100);
      }
      data.push(row);
    }
    
    return {
      data,
      shape,
      device: 'cpu',
      requires_grad: false
    };
  }

  // Reset methods
  resetLIME(): void {
    this.limeData.set(null);
    this.limeState.set(RequestState.Not_Initiated);
  }

  resetSHAP(): void {
    this.shapData.set(null);
    this.shapState.set(RequestState.Not_Initiated);
  }

  resetAll(): void {
    this.resetLIME();
    this.resetSHAP();
    this.loadModelInfo();
    this.loadDatasetInfo();
  }
}
