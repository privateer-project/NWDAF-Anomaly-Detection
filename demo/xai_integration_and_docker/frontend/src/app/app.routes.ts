import { Routes } from '@angular/router';
import { HomeComponent } from './home/home.component';
import { SettingsComponent } from './settings/settings.component';
import { LimeComponent } from './lime/lime.component';
import { ProductionSettingsComponent } from './production-settings/production-settings.component';
import { FeaturesComponent } from './xai/features/features.component';
import { FeaturesComponent as shapLiveFeaturesComponent} from './xai-live/shap/features/features.component' ;
import { TimeseriesComponent } from './xai/timeseries/timeseries.component';
import { TimeseriesComponent as shapLiveTimeseriesComponent  } from './xai-live/shap/timeseries/timeseries.component';
import { ClassificationOutputComponent } from './instance-analysis/classification-output/classification-output.component';
import { WindowComponent } from './xai/window/window.component';
import { WindowComponent as shapLiveWindowComponen} from './xai-live/shap/window/window.component';

export const routes: Routes = [
    {path: '', redirectTo: '/home', pathMatch: 'full'},
    {path: 'home', component: HomeComponent},
    {path: 'settings', component: SettingsComponent},
    {path: 'lime', component: LimeComponent},
    {path: 'production_settings', component: ProductionSettingsComponent},
    {path: 'instance_analysis', component: ClassificationOutputComponent},
    {path: 'xai/timeseries', component: TimeseriesComponent},
    {path: 'xai/features', component: FeaturesComponent},
    {path: 'xai/window', component: WindowComponent},
    {path: 'xai/shap/timeseries', component: shapLiveTimeseriesComponent},
    {path: 'xai/shap/features', component: shapLiveFeaturesComponent},
    {path: 'xai/shap/window', component: shapLiveWindowComponen},
];
