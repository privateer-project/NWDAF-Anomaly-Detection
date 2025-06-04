import { Component } from '@angular/core';

import { FormsModule } from '@angular/forms';
import { HttpClient, HttpClientModule } from '@angular/common/http';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';

@Component({
  selector: 'app-lime',
  standalone: true,
  imports: [FormsModule, HttpClientModule],
  templateUrl: './lime.component.html',
  styleUrl: './lime.component.css'
})
export class LimeComponent {
  instanceValue = 0;
  barGraphicUrl = '';
  tableGraphicUrl = '';
  htmlExplanationUrl = '';
  htmlExplanationUrlSafe: SafeResourceUrl = '';
  noGraphicsMessage = '';

  availableFiles: string[] = [];
  showFileList = false;

  constructor(private http: HttpClient, private sanitizer: DomSanitizer) {
    this.fetchAvailableFiles();
  }

  updateLimeGraphics(): void {
    const instance = this.instanceValue;
    const base = 'http://127.0.0.5:5000/api_lime/files';

    const barUrl = `${base}/lime_bar_classification_instance_${instance}.png`;
    const tableUrl = `${base}/lime_table_classification_instance_${instance}.png`;
    const htmlUrl = `${base}/lime_explanation_classification_instance_${instance}.html`;

    this.noGraphicsMessage = 'Loading...';
    this.barGraphicUrl = '';
    this.tableGraphicUrl = '';
    this.htmlExplanationUrl = '';
    this.htmlExplanationUrlSafe = '';

    this.http.head(barUrl, { observe: 'response' }).subscribe({
      next: (res) => res.status === 200 && (this.barGraphicUrl = barUrl)
    });

    this.http.head(tableUrl, { observe: 'response' }).subscribe({
      next: (res) => res.status === 200 && (this.tableGraphicUrl = tableUrl)
    });

    this.http.head(htmlUrl, { observe: 'response' }).subscribe({
      next: (res) => {
        if (res.status === 200) {
          this.htmlExplanationUrl = htmlUrl;
          this.htmlExplanationUrlSafe = this.sanitizer.bypassSecurityTrustResourceUrl(htmlUrl);
        }
      }
    });

    setTimeout(() => {
      if (!this.barGraphicUrl && !this.tableGraphicUrl && !this.htmlExplanationUrl) {
        this.noGraphicsMessage = 'No graphics found for the selected instance.';
      } else {
        this.noGraphicsMessage = '';
      }
    }, 1000);
  }

  fetchAvailableFiles(): void {
    const url = 'http://127.0.0.5:5000/api_lime/files';
    this.http.get<{ files: string[] }>(url).subscribe({
      next: (res) => this.availableFiles = res.files || [],
      error: () => this.noGraphicsMessage = 'Error loading available files.'
    });
  }

  toggleFileList(): void {
    this.showFileList = !this.showFileList;
  }
}
