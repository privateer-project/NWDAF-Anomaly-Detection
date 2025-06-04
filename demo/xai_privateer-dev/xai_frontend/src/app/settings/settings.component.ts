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


  constructor(private dataService: DataApiService, public shapService: ShapApiService, public limeService: LimeApiService) {
    
   }

   ngOnInit(){
    this.fetchDbNames()
    this.fetchModelNames()
   }

  /* Fetch dataset names from the backend */
  fetchDbNames() {
    this.dataService.listDatasets().subscribe({
      next: (data) => {
        this.dbNames = data.files;
        console.log('Datasets fetched:', this.dbNames);  // Verifique se os dados estão corretos
      },
      error: (err) => console.error('Error fetching dataset names:', err),
    });
  }

  fetchModelNames() {
    this.dataService.listModels().subscribe({
      next: (data) => {
        this.modelNames = data.files;
        console.log('Models fetched:', this.modelNames);  // Verifique se os dados estão corretos
      },
      error: (err) => console.error('Error fetching model names:', err),
    });
  }

  loadAllData() {
    this.loadDataSateMessage = "⌛ Loading all data..."
    let req1 = this.shapService.loadSHAPData(this.datasetName, this.modelName)
    let req2 = this.limeService.loadLIMEData(this.datasetName, this.modelName)
    forkJoin([req1,req2]).subscribe({
      next: (data)=>{
        this.loadDataSateMessage = "✅ All data loaded successfully!"
      },
      error:(err)=>{
        this.loadDataSateMessage = "❌ Error loading data. Please try again."
      }
    })
  }

  loadInstance() {
    this.performCalculationMessage = "⌛ Calculations in progress..."
    let req1 = this.shapService.calculateSHAPelyValues(this.instance)
    let req2 = this.limeService.calculateFeatureImportance(this.instance)
    forkJoin([req1,req2]).subscribe({
      next: (data)=>{
        this.performCalculationMessage = "✅ Instance calculations completed successfully!"
      },
      error:(err)=>{
        this.performCalculationMessage = "❌ Error loading instance. Please try again."
      }
    })
  }

  generateAllGraphics() {
    this.genereateReportsMessage = '⌛ Generating graphics...';
    let req1 = this.shapService.generateSHAPGraphics()
    let req2 = this.limeService.generateLIMEGraphics()
    forkJoin([req1,req2]).subscribe({
      next: (data)=>{
        this.genereateReportsMessage = "✅ Graphics generated successfully!"
      },
      error:(err)=>{
        this.genereateReportsMessage = "❌ Error generating graphics. Please try again."
      }
    })  
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

}
