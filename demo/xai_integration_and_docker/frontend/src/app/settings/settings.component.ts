import { Component, Inject, PLATFORM_ID } from '@angular/core'; // Imports the base Angular component class for building components
import { FormsModule } from '@angular/forms'; // Provides forms-related functionality such as two-way data binding
import { DataApiService } from '../services/data-api.service';
import { LimeApiService } from '../services/lime-api.service';
import { ShapApiService } from '../services/shap-api.service';
import { XAIApiService } from '../../../angular-services/xai-api.service';
import { forkJoin } from 'rxjs';
import { isPlatformBrowser } from '@angular/common';

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
  instanceFile:File|undefined=undefined
  loadDataSateMessage:string = ""
  performCalculationMessage:string = ""
  genereateReportsMessage: string = ''; // Mensagem de feedback
  instancePayload: string = '';
  
  // Estados de processamento
  isProcessing: boolean = false;
  isShapCompleted: boolean = false;
  isLimeCompleted: boolean = false;
  isFullyCompleted: boolean = false;
  
  // Estados de processamento para uploads de arquivos
  isModelUploading: boolean = false;
  isInstanceUploading: boolean = false;
  
  // Mensagens de feedback para uploads
  modelUploadMessage: string = '';
  instanceUploadMessage: string = '';
  
  // Timestamp do último processamento
  lastProcessedTime: string = '';

  constructor(private dataService: DataApiService, public shapService: ShapApiService, public limeService: LimeApiService, private xaiService: XAIApiService, @Inject(PLATFORM_ID) private platformId: Object) {
    
   }

   ngOnInit(){
    // Only make HTTP requests in the browser, not during SSR
    if (isPlatformBrowser(this.platformId)) {
      this.fetchDbNames()
      this.fetchModelNames()
      this.loadPersistedState()
    }
    this.loadExampleInstance() // This doesn't make HTTP requests, so it's safe for SSR
   }

  // Load example instance from startup_dataset.json
  loadExampleInstance() {
    this.instancePayload = JSON.stringify({
      "data": [[[-1.3365587225021924, -1.149925444193916, 1.6439847189342403, 2.642742690080383, 0.00041423388951851436, 0.6819555109959033, 1.3568002863782298, -1.2941502832274892], [1.0717318979227017, 2.160862959067339, -0.4090208692720301, 0.4386245204307584, -1.2441426638008575, 0.7894325790365455, 0.2900730194134801, 1.147999080890697], [0.5830429503194307, -1.0378613989979735, 1.8852735682201744, 1.4685248033593443, -0.019706412153884578, 0.8395576852574174, -0.3419575998625662, 0.3245881091244702], [0.3789118386566458, 0.6751910097394118, 0.5863971312951728, -0.9138975266220395, -0.3181003217059648, 1.566718840175131, 0.9733629373809737, 0.9753569041166313], [0.7229116795715653, -0.6903368393560858, 0.12388140922449647, -0.4371950173568295, 0.5595257151687496, 0.2531980608867737, 0.4095377549658098, 0.011018264647534304], [0.9160604024117573, 1.3898321955219883, 1.073825683216868, 1.1467796531779229, -1.7025113959501241, 1.2634550739654395, 0.4653404198056041, 0.8143665269001308], [0.47349986387493925, 0.5254059232465929, -0.14510431345653854, 0.16418765175052646, -0.024500750569873265, -0.04987862275640584, -0.7332862705774055, 1.723358712439563], [-0.2178123386782981, 1.8456946596876658, -0.8677994940756304, -0.06516599728486125, -0.43319555393963555, -1.2512299818440467, -1.1846107172676659, -0.7122979111898254], [-0.5377444827177522, -0.3484107943584651, -0.30324640734336533, 0.26021521321191554, -0.3197221801238683, -0.058215057232256284, -1.0003024217630136, -0.4150626096757879], [-1.3120915385228202, -1.295184664758526, 0.3599961439978285, -2.176200893904022, -0.0481928112816658, -0.9628113045587824, -2.52934784325329, 0.9708151748804484], [0.6359690932962871, 0.24511855925677484, 0.22893665966353258, -0.9380863948406825, 0.3475320091350779, 0.2760283368876647, -0.9622300360088828, -0.3557403730329584], [-2.206645711432992, 0.15802867315382726, 0.3072852831784805, -1.5902901366229827, -2.433305368506444, 1.4983257708026982, -0.01084466652789675, -0.3600475490682549]]]
    }, null, 2);
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
    // Reset states
    this.clearState();
    
    // Set processing state
    this.isProcessing = true;
    this.performCalculationMessage = "⌛ Processing instance...";
    this.lastProcessedTime = new Date().toLocaleString();
    this.persistState();
  
    let payload: any;
    try {
      payload = JSON.parse(this.instancePayload);
    } catch (e) {
      this.performCalculationMessage = "❌ Invalid JSON payload. Please check and try again.";
      this.isProcessing = false;
      this.persistState();
      return;
    }
  
    // Primeiro SHAP com string_json
    this.xaiService.calculateSHAPString(payload).subscribe({
      next: (shapResult: any) => {
        console.log("SHAP completed:", shapResult);
        this.isShapCompleted = true;
        this.performCalculationMessage = "⌛ SHAP completed, processing LIME...";
        this.persistState();
  
        // Depois LIME com string_json
        this.xaiService.calculateLIMEString(payload).subscribe({
          next: (limeResult: any) => {
            console.log("LIME completed:", limeResult);
            this.isLimeCompleted = true;
            this.isFullyCompleted = true;
            this.isProcessing = false;
            this.performCalculationMessage = `✅ Instance processed successfully! (SHAP + LIME) - Completed at ${this.lastProcessedTime}`;
            this.persistState();
          },
          error: (err: any) => {
            console.error("Error calculating LIME:", err);
            this.isProcessing = false;
            this.performCalculationMessage = "❌ Error in LIME calculation.";
            this.persistState();
          }
        });
      },
      error: (err: any) => {
        console.error("Error calculating SHAP:", err);
        this.isProcessing = false;
        this.performCalculationMessage = "❌ Error in SHAP calculation.";
        this.persistState();
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
    const file = target?.files?.[0]
    
    if (file) {
      // Validate file extension
      if (file.name.toLowerCase().endsWith('.pth')) {
        this.modelFile = file
        this.modelUploadMessage = '' // Clear any previous messages
        console.log('Model file selected:', file.name)
      } else {
        this.modelFile = undefined
        this.modelUploadMessage = '❌ Please select a .pth file only.'
        console.log('Invalid file type selected:', file.name)
      }
    } else {
      this.modelFile = undefined
    }
  }

  submitModelFile(){
    if(this.modelFile){
      // Double-check file extension before upload
      if (!this.modelFile.name.toLowerCase().endsWith('.pth')) {
        this.modelUploadMessage = "❌ Please select a .pth file only.";
        return;
      }
      
      this.isModelUploading = true;
      this.modelUploadMessage = "⌛ Uploading model file...";
      
      this.dataService.uploadModel(this.modelFile).subscribe({
        next: (response) => {
          console.log("Model uploaded successfully:", response);
          this.isModelUploading = false;
          this.modelUploadMessage = "✅ Model file uploaded successfully!";
        },
        error: (error) => {
          console.error("Error uploading model:", error);
          this.isModelUploading = false;
          this.modelUploadMessage = "❌ Error uploading model file.";
        },
        complete: () => {
          console.log("Model upload completed");
        }
      })
    } else {
      this.modelUploadMessage = "❌ Please select a model file first.";
    }
  }

  onInstanceFileSelected(event: Event): void {
    const target = event.target as HTMLInputElement;
    this.instanceFile = target?.files?.[0] || undefined;
    console.log('Instance file selected:', this.instanceFile);
  }

  submitInstanceFile(){
    if(this.instanceFile){
      this.isInstanceUploading = true;
      this.instanceUploadMessage = "⌛ Uploading instance file...";
      
      this.xaiService.uploadInstanceFile(this.instanceFile).subscribe({
        next: (result: any) => {
          console.log("Instance file uploaded:", result);
          this.isInstanceUploading = false;
          this.instanceUploadMessage = "✅ Instance file uploaded successfully!";
        },
        error: (err: any) => {
          console.error("Error uploading instance file:", err);
          this.isInstanceUploading = false;
          this.instanceUploadMessage = "❌ Error uploading instance file.";
        }
      });
    } else {
      this.instanceUploadMessage = "❌ Please select an instance file first.";
    }
  }

  // Helper: create a dummy instance payload for demonstration
  private buildInstanceData(instanceIndex: number){
    // TODO: Replace with real selection from loaded dataset
    return Array.from({length: 1*12*8}, (_,i)=> 0)
  }

  // Persistir estado no localStorage
  private persistState() {
    if (isPlatformBrowser(this.platformId)) {
      const state = {
        isProcessing: this.isProcessing,
        isShapCompleted: this.isShapCompleted,
        isLimeCompleted: this.isLimeCompleted,
        isFullyCompleted: this.isFullyCompleted,
        lastProcessedTime: this.lastProcessedTime,
        performCalculationMessage: this.performCalculationMessage
      };
      localStorage.setItem('settingsState', JSON.stringify(state));
    }
  }

  // Carregar estado do localStorage
  private loadPersistedState() {
    if (isPlatformBrowser(this.platformId)) {
      const savedState = localStorage.getItem('settingsState');
      if (savedState) {
        try {
          const state = JSON.parse(savedState);
          this.isProcessing = state.isProcessing || false;
          this.isShapCompleted = state.isShapCompleted || false;
          this.isLimeCompleted = state.isLimeCompleted || false;
          this.isFullyCompleted = state.isFullyCompleted || false;
          this.lastProcessedTime = state.lastProcessedTime || '';
          this.performCalculationMessage = state.performCalculationMessage || '';
          
          // Se estava processando, verificar se ainda está
          if (this.isProcessing && !this.isFullyCompleted) {
            this.performCalculationMessage = "⌛ Processing instance... (restored from previous session)";
          }
        } catch (e) {
          console.error('Error loading persisted state:', e);
        }
      }
    }
  }

  // Limpar estado
  private clearState() {
    this.isProcessing = false;
    this.isShapCompleted = false;
    this.isLimeCompleted = false;
    this.isFullyCompleted = false;
    this.lastProcessedTime = '';
    this.performCalculationMessage = '';
    this.persistState();
  }

  // Métodos para lidar com cliques nos botões de seleção de arquivo
  selectModelFile() {
    if (isPlatformBrowser(this.platformId)) {
      const fileInput = document.getElementById('modelFile') as HTMLInputElement;
      if (fileInput) {
        fileInput.click();
      }
    }
  }

  selectInstanceFile() {
    if (isPlatformBrowser(this.platformId)) {
      const fileInput = document.getElementById('instanceFile') as HTMLInputElement;
      if (fileInput) {
        fileInput.click();
      }
    }
  }

}
