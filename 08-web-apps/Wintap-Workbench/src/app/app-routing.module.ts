import { RouterModule } from '@angular/router';
import { NgModule } from '@angular/core';
import { NotfoundComponent } from './notfound/notfound.component';
import { AppLayoutComponent } from "./layout/app.layout.component";
import { QuerybuilderComponent } from './querybuilder/querybuilder.component';

@NgModule({
    imports: [
        RouterModule.forRoot([
            {
                path: '', component: AppLayoutComponent,
                children: [
                    { path: '', loadChildren: () => import('./dashboard/dashboard.module').then(m => m.DashboardModule) },
                    { path: 'querybuilder', component: QuerybuilderComponent },
                    { path: 'documentation', loadChildren: () => import('./documentation/documentation.module').then(m => m.DocumentationModule) },
                ]
            },
            { path: 'notfound', component: NotfoundComponent },
            { path: '**', redirectTo: '/notfound' },
        ], { scrollPositionRestoration: 'enabled', anchorScrolling: 'enabled', onSameUrlNavigation: 'reload' })
    ],
    exports: [RouterModule]
})
export class AppRoutingModule {
}
