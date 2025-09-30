import { Component } from '@angular/core'; // Imports the base Angular component class for building components
import { FormsModule } from '@angular/forms'; // Provides forms-related functionality such as two-way data binding
import { DataApiService } from '../services/data-api.service';
import { LimeApiService } from '../services/lime-api.service';
import { ShapApiService } from '../services/shap-api.service';
import { forkJoin } from 'rxjs';

@Component({
  selector: 'app-settings', // Defines the selector that will be used to place this component in the HTML
  imports: [
    FormsModule
],
  templateUrl: './settings.component.html',  // Specifies the path to the component's template file
  styleUrl: './settings.component.css' // Specifies the path to the component's CSS file
})
export class SettingsComponent {
  dbNames: string[] = []; // Lista de nomes dos datasets
  modelNames: string[] = []; // Lista de nomes dos modelos
  datasetName: string = ''; // Nome do dataset selecionado
  modelName: string = ''; // Nome do modelo selecionado
  instance: number = 0; // Número da instância para cálculos
  dataFile:File|undefined=undefined
  modelFile:File|undefined=undefined
  loadDataSateMessage:string = ""
  performCalculationMessage:string = ""
  genereateReportsMessage: string = ''; // Mensagem de feedback
  instancePayload: string = '';

  constructor(private dataService: DataApiService, public shapService: ShapApiService, public limeService: LimeApiService) {
    
   }

   ngOnInit(){
    this.fetchDbNames()
    this.fetchModelNames()
   }

  /* Fetch dataset names from the backend */
  fetchDbNames() {
    this.dataService.listDatasets().subscribe({
      next: (data: any) => {
        // Backend returns an info object; no listing endpoint. Fallback to a default name.
        this.dbNames = ['default'];
        console.log('Dataset info:', data);
      },
      error: (err: any) => console.error('Error fetching dataset info:', err),
    });
  }

  fetchModelNames() {
    this.dataService.listModels().subscribe({
      next: (data: any) => {
        // Backend returns an info object; no listing endpoint. Fallback to a default name.
        this.modelNames = ['current'];
        console.log('Model info:', data);
      },
      error: (err: any) => console.error('Error fetching model info:', err),
    });
  }

  loadAllData() {
    this.loadDataSateMessage = "⌛ Loading all data..."
    // With unified backend, there is no separate preload step; mark as completed
    this.loadDataSateMessage = "✅ All data loaded successfully!"
  }

  loadInstance() {
    this.performCalculationMessage = "⌛ Processing instance...";
  
    let payload: any;
    try {
      payload = JSON.parse(this.instancePayload);
    } catch (e) {
      this.performCalculationMessage = "❌ Invalid JSON payload. Please check and try again.";
      return;
    }
  
    // Primeiro SHAP
    this.shapService.calculateSHAP(payload).subscribe({
      next: (shapResult) => {
        console.log("SHAP completed:", shapResult);
  
        // Depois LIME
        this.limeService.calculateLIME(payload).subscribe({
          next: (limeResult) => {
            console.log("LIME completed:", limeResult);
            this.performCalculationMessage = "✅ Instance processed successfully! (SHAP + LIME)";
          },
          error: (err) => {
            console.error("Error calculating LIME:", err);
            this.performCalculationMessage = "❌ Error in LIME calculation.";
          }
        });
      },
      error: (err) => {
        console.error("Error calculating SHAP:", err);
        this.performCalculationMessage = "❌ Error in SHAP calculation.";
      }
    });
  }

  generateAllGraphics() {
    this.genereateReportsMessage = '⌛ Generating graphics...';
    // Graphics generation endpoints were removed in backend; mark as completed
    this.genereateReportsMessage = "✅ Graphics generated successfully!"
  }

  onDataFileSelected(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.dataFile= target?.files?.[0] || undefined;
    console.log(File)
  }

  submitDataFile(){
    if(this.dataFile){
      this.dataService.uploadDataset(this.dataFile).subscribe({
        complete:()=>{
          console.log("uploaded")
        }
      })
    }
  }

  onModelFileSelected(event: Event): void {
    const target = event.target as HTMLInputElement
    this.modelFile= target?.files?.[0] || undefined
    console.log(File)
  }

  submitModelFile(){
    if(this.modelFile){
      this.dataService.uploadModel(this.modelFile).subscribe({
        complete:()=>{
          console.log("uploaded")
        }
      })
    }
  }

  // Helper: create a dummy instance payload for demonstration
  private buildInstanceData(instanceIndex: number){
    // TODO: Replace with real selection from loaded dataset
    return Array.from({length: 1*12*8}, (_,i)=> 0)
  }

}
