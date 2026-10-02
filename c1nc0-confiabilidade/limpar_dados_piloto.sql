-- C1NC0 - limpar dados do piloto antes da coleta oficial
-- Execute no SQL Editor do mesmo projeto Supabase usado pelo piloto.
-- A ordem abaixo respeita a FK de c1nc0_feedback.analise_id.

begin;

delete from public.c1nc0_feedback;
delete from public.c1nc0_analises;

commit;

-- Conferência: ambos devem retornar 0.
select count(*) as feedbacks_restantes from public.c1nc0_feedback;
select count(*) as analises_restantes from public.c1nc0_analises;
