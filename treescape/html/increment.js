Vue.component("my-component", {
    template: '<div>' +
        '<p>Hello from Vue component!</p>' +
        '<button @click="increment">Increment</button>' +
        '<p>Count: {{ count }}</p></div>',
    data: () => ({
      count: 0
    }),
    methods: {
        increment() {
            this.count++
        }
    }
});

new Vue({
    el: '#app',
    data: {
        initialCount: 5
    }
});
