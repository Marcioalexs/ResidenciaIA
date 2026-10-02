create extension if not exists pgcrypto;

create table if not exists public.c1nc0_analises (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    session_id uuid not null,
    origem text not null check (origem in ('site','extensao')),
    modo_modelo text not null check (modo_modelo in ('gaussian','multinomial','comparar')),
    modelos_concordaram boolean,
    versao_app text not null default 'piloto-2026-10-03'
);

create table if not exists public.c1nc0_feedback (
    id uuid primary key default gen_random_uuid(),
    created_at timestamptz not null default now(),
    session_id uuid not null,
    analise_id uuid references public.c1nc0_analises(id) on delete set null,
    origem text not null check (origem in ('site','extensao')),
    modo_modelo text not null check (modo_modelo in ('gaussian','multinomial','comparar')),
    modelos_concordaram boolean,
    ajudou_analise integer check (ajudou_analise between 1 and 5),
    buscaria_outras_fontes text check (buscaria_outras_fontes in ('sim','talvez','nao')),
    clareza_limites integer check (clareza_limites between 1 and 5),
    clareza_comparacao_modelos integer check (clareza_comparacao_modelos between 1 and 5),
    mudou_avaliacao text check (mudou_avaliacao in ('sim','nao','ainda_duvidas')),
    parte_mais_util text check (parte_mais_util in ('observamos','nao_sabemos','o_que_checar','leitura_lateral','ml')),
    usaria_novamente text check (usaria_novamente in ('sim','talvez','nao')),
    comentario text,
    versao_app text not null default 'piloto-2026-10-03'
);

create index if not exists idx_c1nc0_analises_created_at on public.c1nc0_analises(created_at);
create index if not exists idx_c1nc0_analises_session on public.c1nc0_analises(session_id);
create index if not exists idx_c1nc0_feedback_created_at on public.c1nc0_feedback(created_at);
create index if not exists idx_c1nc0_feedback_session on public.c1nc0_feedback(session_id);

alter table public.c1nc0_analises enable row level security;
alter table public.c1nc0_feedback enable row level security;

revoke all on public.c1nc0_analises from anon, authenticated;
revoke all on public.c1nc0_feedback from anon, authenticated;
grant select, insert on public.c1nc0_analises to service_role;
grant select, insert on public.c1nc0_feedback to service_role;
