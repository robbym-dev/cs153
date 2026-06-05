export interface ScopeItem {
  code: string;
  quantity: number;
  unit: string;
  description: string;
}

export interface LineItem {
  code: string;
  description: string;
  quantity: number;
  unit: string;
  unit_cost: number;
  unit_labor: number;
  unit_material: number;
  total: number;
}

export interface BidSummary {
  subtotal: number;
  markups: {
    overhead: number;
    tax: number;
    bid_bond: number;
    contingencies: number;
  };
  grand_total: number;
}

export type AlertSeverity = "critical" | "warning" | "info";

export interface ScopeAlert {
  item_id: string;
  severity: AlertSeverity;
  description: string;
  suggested_action: string;
}

export interface BidResponse {
  scope_items: ScopeItem[];
  line_items: LineItem[];
  summary: BidSummary;
  alerts: ScopeAlert[];
  download_url: string;
}

export type Stage =
  | "extracting"
  | "pricing"
  | "scope_check"
  | "generating_excel"
  | "done"
  | "error";

export interface ProgressEvent {
  stage: Stage;
  current?: number;
  total?: number;
  page?: number;
  warning?: string;
  message?: string;
  result?: BidResponse;
}

export interface ReferenceLine {
  code: string;
  unit: string;
  qty: number;
  total: number;
}

export interface ReferenceResponse {
  filename: string;
  lines: ReferenceLine[];
  aggregate_total: number;
}
