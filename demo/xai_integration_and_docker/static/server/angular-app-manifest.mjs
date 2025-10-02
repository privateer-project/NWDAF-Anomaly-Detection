
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
    'index.csr.html': {size: 23613, hash: '5ebcf41af5c8273a4951b15e6a95d78a9833af5bc970fd3dedfb6ca68e1be6a1', text: () => import('./assets-chunks/index_csr_html.mjs').then(m => m.default)},
    'index.server.html': {size: 17187, hash: 'e5970c4d95616d76186882398974f21c1ca0766c0abaa0f443077aed5ee14b9a', text: () => import('./assets-chunks/index_server_html.mjs').then(m => m.default)},
    'settings/index.html': {size: 38437, hash: '5779921c6e186fab12d7a343f686769d3b096cc18673c92ee961962327a773f3', text: () => import('./assets-chunks/settings_index_html.mjs').then(m => m.default)},
    'shap8/index.html': {size: 29647, hash: '08b87413a793d0b0a2cdbc221e6d6f6494709462323076eb37eb8782703be85f', text: () => import('./assets-chunks/shap8_index_html.mjs').then(m => m.default)},
    'lime/index.html': {size: 30223, hash: '1ee102d629e81f3d581e636b2d991effa3a066c28d1cdd20e582361bc0ab50df', text: () => import('./assets-chunks/lime_index_html.mjs').then(m => m.default)},
    'production_settings/index.html': {size: 32786, hash: '09f59a1eaf2affdc6bbfd5407f5bed1b2017af9a06d5b94066bc7ea79e6e9290', text: () => import('./assets-chunks/production_settings_index_html.mjs').then(m => m.default)},
    'shap1/index.html': {size: 29402, hash: '65e08a14f39a03608bcea8ca77e41ffd7b2983a7442a7b1483360070ff9b4de2', text: () => import('./assets-chunks/shap1_index_html.mjs').then(m => m.default)},
    'xai/timeseries/index.html': {size: 29585, hash: 'c0ea95b0fab4fa4388e70d7a7d9186e5eea76a900619459871f96a872f8aa2fd', text: () => import('./assets-chunks/xai_timeseries_index_html.mjs').then(m => m.default)},
    'instance_analysis/index.html': {size: 40804, hash: '68e9360d80683ced9237ea59c405f991f77dea9c12c27c7ffbdcd2ad868b66e8', text: () => import('./assets-chunks/instance_analysis_index_html.mjs').then(m => m.default)},
    'xai/features/index.html': {size: 26531, hash: '654e44820f8eb671d27f851271ef5f8ebf3a3361c4cac5bac2734e504cbddbcc', text: () => import('./assets-chunks/xai_features_index_html.mjs').then(m => m.default)},
    'xai/window/index.html': {size: 26525, hash: 'f4fd966ac67ae776bf2e6c9c6f898fca02eeb825bf67d9d9056a68c91465e095', text: () => import('./assets-chunks/xai_window_index_html.mjs').then(m => m.default)},
    'home/index.html': {size: 30479, hash: 'e5e1f3e70997e31b828935e3a4b6d7e6d8f28d1769589b87a93d5d72c895f8d8', text: () => import('./assets-chunks/home_index_html.mjs').then(m => m.default)},
    'styles-CXQUZ3PB.css': {size: 6979, hash: 'mYIPdabeAag', text: () => import('./assets-chunks/styles-CXQUZ3PB_css.mjs').then(m => m.default)}
  },
};
