import {
    Component, ViewChild, AfterViewInit, OnInit, ElementRef, QueryList
} from '@angular/core';
import { Table } from 'primeng/table';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { ChangeDetectorRef } from '@angular/core';
import 'codemirror/mode/sql/sql';
import { CodemirrorComponent } from '@ctrl/ngx-codemirror';
import * as CodeMirror from 'codemirror';
import 'codemirror/addon/mode/overlay';
import './wintapmessage';
import 'signalr';

declare var $: any;

interface expandedRows {
    [key: string]: boolean;
}

@Component({
  selector: 'app-querybuilder',
  templateUrl: './querybuilder.component.html',
    styleUrls: ['./querybuilder.component.css']
})
export class QuerybuilderComponent implements AfterViewInit, OnInit {

    ngAfterViewInit(): void {
        configureCodeMirror();
        this.fetchEplListing();
    }

    @ViewChild('editor') codeEditor!: CodemirrorComponent;
    @ViewChild('nameField') nameField!: ElementRef;
    @ViewChild('dt1') table!: Table;
    @ViewChild('resultsContainer') resultsContainer!: ElementRef;


    eplListing: Statement[] = [];
    selectedStatement: any = null;
    loading: boolean = true;
    showConfirmDialog = false;
    showInvalidQueryDialog = false;
    private connection: any;
    queryResults: string[] = [];
    queryError: string = '';

    constructor(private http: HttpClient, private cd: ChangeDetectorRef) {
        this.connection = $.hubConnection('/signalr');
        const hubProxy = this.connection.createHubProxy('workbenchHub');
      
        // Add event handlers for the hub
        hubProxy.on('addMessage', (data: any) => {
          this.queryResults.push(decodeURIComponent(data));
          this.cd.detectChanges();
          setTimeout(() => this.scrollToBottom(), 0);
        });
      
        // Start the connection
        this.connection.start()
          .done(() => {
            console.log('Connected to the hub');
          })
          .fail((error: any) => {
            console.error('Failed to connect to the hub:', error);
          });
    }

    ngOnInit() {
        this.loading = false;
    }

    onSort() {
        // TODO
    }

    onGlobalFilter(table: Table, event: Event) {
        table.filterGlobal((event.target as HTMLInputElement).value, 'contains');
    }

    clear(table: Table) {
        this.deleteEpl() ;
    }

    activateEpl(eplName: string) {
        this.queryResults = [];
        const codeMirrorInstance = this.codeEditor.codeMirror;
        const editorContent = codeMirrorInstance!.getValue();
        const encodedContent = encodeURIComponent(editorContent);
        this.addStream(eplName, encodedContent, "ACTIVE").subscribe(
            (response) => {
                console.log('Success:', response);
                this.fetchEplListing();
            },
            (error) => {
                console.log('Error:' + JSON.stringify(error));
                this.queryError = error.error.Message;;
                this.showInvalidQueryDialog = true;
            }
        );
    }

    startEpl() {
        const selectedRow = this.table.selection;
        const eplName = selectedRow.Name;
        const queryString = selectedRow.Query;
        const encodedContent = encodeURIComponent(queryString);
        this.addStream(eplName, encodedContent, "START").subscribe(
          (response) => {
              console.log('Success:', response);
              this.fetchEplListing();
          },
          (error) => {
              console.log('Error:' + JSON.stringify(error));
              this.queryError = error.error.Message;;
              this.showInvalidQueryDialog = true;
          }
      );
    }

    stopEpl() {
        const selectedRow = this.table.selection;
        const eplName = selectedRow.Name;
        const queryString = selectedRow.Query;
        const encodedContent = encodeURIComponent(queryString);
        this.addStream(eplName, encodedContent, "STOP").subscribe(
          (response) => {
              console.log('Success:', response);
              this.fetchEplListing();
          },
          (error) => {
              console.log('Error:' + JSON.stringify(error));
              this.queryError = error.error.Message;;
              this.showInvalidQueryDialog = true;
          }
      );
    }

    deleteOneEpl() {
      const selectedRow = this.table.selection;
      const eplName = selectedRow.Name;
      const queryString = selectedRow.Query;
      const encodedContent = encodeURIComponent(queryString);
      this.addStream(eplName, encodedContent, "DELETE").subscribe(
        (response) => {
            console.log('Success:', response);
            this.fetchEplListing();
        },
        (error) => {
            console.log('Error:' + JSON.stringify(error));
            this.queryError = error.error.Message;;
            this.showInvalidQueryDialog = true;
        }
    );
  }

    editEpl() {
        const selectedRow = this.table.selection;
        const name = selectedRow.Name;
        const queryString = selectedRow.Query;
        var decodedString = queryString;
        try{
            decodedString = decodeURIComponent(queryString);
        }
        catch(error) {
            console.log(error);
        }
        this.codeEditor.codeMirror!.setValue(decodedString);
        this.nameField.nativeElement.value = name;
    }

    deleteEpl() {  
        this.confirmDelete(); // Display the confirmation dialog
    }

    deleteConfirmed() {
        const apiUrl = `/api/Streams/`;
        this.http.delete(apiUrl).toPromise()
          .then(() => {
            this.fetchEplListing();
            this.showConfirmDialog = false;
            this.cd.detectChanges();
          });
      }

    confirmDelete() {
        this.showConfirmDialog = true;
      }

    addStream(shortName: string, queryString: string, stateString: string): Observable<any> {
        const apiUrl = `/api/Streams?name=${shortName}&query=${queryString}&state=${stateString}`;
        return this.http.post(apiUrl, null);
    }

    fetchEplListing() {
        this.http.get<ApiResponse>('/api/Streams').subscribe(response => {
            this.eplListing = response.response;
            console.log(JSON.stringify(this.eplListing))
        });
    }

    private scrollToBottom() {
        const container = this.resultsContainer.nativeElement;
        container.scrollTop = container.scrollHeight;
      }
      
}

function configureCodeMirror() {
    CodeMirror.defineMode('wintapOverlay', (config, parserConfig) => {
      const esperMode = CodeMirror.getMode(config, 'text/x-esper');
      const myOverlayMode = CodeMirror.getMode(config, 'wintapmessage');
      return CodeMirror.overlayMode(esperMode, myOverlayMode);
    });
  }
  

export interface ApiResponse {
    response: Statement[];
}

export interface Statement {
    Name: string;
    Query: string;
    StatementType: string | null;
    State: string | null;
    CreateDate: number;
}

