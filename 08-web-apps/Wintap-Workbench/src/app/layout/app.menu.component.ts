import { OnInit } from '@angular/core';
import { Component } from '@angular/core';
import { LayoutService } from './service/app.layout.service';
import { QuerybuilderComponent } from '../querybuilder/querybuilder.component';

@Component({
    selector: 'app-menu',
    templateUrl: './app.menu.component.html'
})
export class AppMenuComponent implements OnInit {

    model: any[] = [];

    constructor(public layoutService: LayoutService) { }

    ngOnInit() {
        this.model = [
            {
                label: 'Home',
                items: [
                    { label: 'Dashboard', icon: 'pi pi-fw pi-home', routerLink: ['/'] }
                ]
            },
            {
                label: 'Workbench Tools',
                items: [
                    { label: 'Query Builder', icon: 'pi pi-fw pi-database', routerLink: ['/querybuilder'] },
                    // { label: 'ETW Explorer', icon: 'pi pi-fw pi-compass', routerLink: ['/uikit/input'] },
                ]
            },
            {
                label: 'Get Started',
                items: [
                    {
                        label: 'Documentation', icon: 'pi pi-fw pi-question', routerLink: ['/documentation']
                    },
                    {
                        label: 'Code', icon: 'pi pi-fw pi-github', url: ['https://github.com/LLNL/Wintap-Workbench'], target: '_blank'
                    }
                ]
            }
        ];
    }
}
