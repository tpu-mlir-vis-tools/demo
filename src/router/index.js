import { createRouter, createWebHashHistory } from 'vue-router'
import LmemView from '../ui/views/lmem-view.vue'
import TimestepView from '../ui/views/timestep-view.vue'
import ProfileView from '../ui/views/profile-view.vue'
import NeuronView from '../ui/views/neuron-view.vue'


const routes = [
  {
    path: '/',
    redirect: '/lmem'
  },
  {
    path: '/lmem',
    name: 'lmem',
    component: LmemView
  },
  {
    path: '/timestep',
    name: 'timestep',
    component: TimestepView
  },
  {
    path: '/profile',
    name: 'profile',
    component: ProfileView
  },
  {
    path: '/neuron',
    name: 'neuron',
    component: NeuronView
  },

]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

export default router