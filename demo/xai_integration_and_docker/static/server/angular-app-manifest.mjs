
export default {
  bootstrap: () => import('./main.server.mjs').then(m => m.default),
  inlineCriticalCss: true,
  baseHref: '/',
  locale: undefined,
  routes: [
  {
    "renderMode": 2,
    "redirectTo": "/home",
    "route": "/"
  },
  {
    "renderMode": 2,
    "route": "/home"
  },
  {
    "renderMode": 2,
    "route": "/settings"
  },
  {
    "renderMode": 2,
    "route": "/shap1"
  },
  {
    "renderMode": 2,
    "route": "/shap8"
  },
  {
    "renderMode": 2,
    "route": "/lime"
  },
  {
    "renderMode": 2,
    "route": "/production_settings"
  },
  {
    "renderMode": 2,
    "route": "/instance_analysis"
  },
  {
    "renderMode": 2,
    "route": "/xai/timeseries"
  },
  {
    "renderMode": 2,
    "route": "/xai/features"
  },
  {
    "renderMode": 2,
    "route": "/xai/window"
  }
],
  entryPointToBrowserMapping: undefined,
  assets: {
    'index.csr.html': {size: 23613, hash: 'fd2b5f18360b835576159d9953223b9af902cffaedca2c5e08e4923de5612b6d', text: () => import('./assets-chunks/index_csr_html.mjs').then(m => m.default)},
    'index.server.html': {size: 17187, hash: 'ff0bef26e8ef8852a5dad4c2bb564fc160255f7ceb25fc43518ced3a1d0593b1', text: () => import('./assets-chunks/index_server_html.mjs').then(m => m.default)},
    'home/index.html': {size: 30479, hash: '48a6d58a513413b14b7893ac50d42ed39a2500289e36e642e2eae591bfb0d6e5', text: () => import('./assets-chunks/home_index_html.mjs').then(m => m.default)},
    'lime/index.html': {size: 30223, hash: '1f90bce900846eed00b7842342c0ca0f3665814ae26bd78a679a2d85a2499e85', text: () => import('./assets-chunks/lime_index_html.mjs').then(m => m.default)},
    'production_settings/index.html': {size: 32786, hash: 'd53db68283e37ffa5dd1546c6a3f3df9adf419330af014c107532c96752af6c5', text: () => import('./assets-chunks/production_settings_index_html.mjs').then(m => m.default)},
    'instance_analysis/index.html': {size: 40804, hash: 'ebf184f07df0ba6e86536c1cc57eb10442022dc96546e55968618433c3e6238d', text: () => import('./assets-chunks/instance_analysis_index_html.mjs').then(m => m.default)},
    'shap1/index.html': {size: 29830, hash: 'b1bacd7c452a9fe1f100a3e46939b01effa99a916a0487ea06491f179c3c0c6a', text: () => import('./assets-chunks/shap1_index_html.mjs').then(m => m.default)},
    'settings/index.html': {size: 31540, hash: '6354c86a55273c08a6fbe79141c70beb94b716c003d59a9265883de564d9ac05', text: () => import('./assets-chunks/settings_index_html.mjs').then(m => m.default)},
    'xai/features/index.html': {size: 26531, hash: '6b320cdc238c595d13eb39a77113a508d996bb3b29fddf5bdc7da69bd55b93bd', text: () => import('./assets-chunks/xai_features_index_html.mjs').then(m => m.default)},
    'xai/window/index.html': {size: 26531, hash: 'd96b5e5a579868c2fb2c860757f97ec87529d284169eefb54cccb30be6720f3c', text: () => import('./assets-chunks/xai_window_index_html.mjs').then(m => m.default)},
    'xai/timeseries/index.html': {size: 53080, hash: 'c5bc42ff05c35a02461ba42f2be052ca54cb5b1aeb2ac1de02fc23a8d3982587', text: () => import('./assets-chunks/xai_timeseries_index_html.mjs').then(m => m.default)},
    'shap8/index.html': {size: 29696, hash: '871ea8cd1f1e2f5a775eb226b6818156c1162463b9cc4f7a72bad5ce838db9b5', text: () => import('./assets-chunks/shap8_index_html.mjs').then(m => m.default)},
    'styles-CXQUZ3PB.css': {size: 6979, hash: 'mYIPdabeAag', text: () => import('./assets-chunks/styles-CXQUZ3PB_css.mjs').then(m => m.default)}
  },
};
