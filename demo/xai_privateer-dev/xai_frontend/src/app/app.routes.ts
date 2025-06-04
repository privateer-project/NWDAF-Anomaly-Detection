import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { SettingsComponent } from './settings/settings.component';
import { Shap1Component } from './shap-1/shap-1.component';
import { Shap8Component } from './shap-8/shap-8.component';
import { LimeComponent } from './lime/lime.component';
import { ProductionSettingsComponent } from './production-settings/production-settings.component';
import { FeaturesComponent } from './xai/features/features.component';
import { TimeseriesComponent } from './xai/timeseries/timeseries.component';
import { ClassificationOutputComponent } from './instance-analysis/classification-output/classification-output.component';
import { WindowComponent } from './xai/window/window.component';

export const routes: Routes = [
    {path: '', redirectTo: '/home', pathMatch: 'full'},
    {path: 'home', component: HomeComponent},
    {path: 'settings', component: SettingsComponent},
    {path: 'shap1', component: Shap1Component},
    {path: 'shap8', component: Shap8Component},
    {path: 'lime', component: LimeComponent},
    {path: 'production_settings', component: ProductionSettingsComponent},
    {path: 'instance_analysis', component: ClassificationOutputComponent},
    {path: 'xai/timeseries', component: TimeseriesComponent},
    {path: 'xai/features', component: FeaturesComponent},
    {path: 'xai/window', component: WindowComponent},
];
