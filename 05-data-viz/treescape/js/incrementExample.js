if( !window.createApp ) {

    const {createApp, ref} = Vue;

    const app = createApp({
        setup() {
            const count = ref(0);
            const increment = () => {
                count.value++;
            };

            return {
                count,
                increment,
            };
        },
    });

    const elements = document.querySelectorAll('.mainApps');

    elements.forEach(el => {
      app.mount(el);
    });
}